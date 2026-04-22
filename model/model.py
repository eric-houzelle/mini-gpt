import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from .configuration import MiniGPTConfig


# ---------------------------------------------------------------------------
# Recurrent-Depth Transformer (RDT) components
# Ref: OpenMythos / Parcae (Prairie et al., 2026)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """Linear Time-Invariant state injection for the recurrent loop.

    At each iteration t:
        h(t+1) = A · h(t) + B · e + transformer_output

    Stability guarantee: ρ(A) < 1 by construction.
    A is parameterised as  A = U · diag(σ(a)) · V^T  where  σ(a) ∈ (0, 1)
    via a sigmoid on unconstrained parameters *a*.  Because the singular
    values are all < 1, the spectral radius is guaranteed < 1.
    We use a *diagonal* A for efficiency (embed_dim params, not embed_dim²).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.a_logit = nn.Parameter(torch.zeros(embed_dim))
        self.b_scale = nn.Parameter(torch.ones(embed_dim) * 0.1)

    def forward(self, h: torch.Tensor, encoded_input: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.a_logit).clamp(max=0.95)
        return a * h + self.b_scale * encoded_input


class DepthLoRA(nn.Module):
    """Low-rank adaptation applied per recurrent depth step.

    Adds a small  delta = (x @ A) @ B  to the hidden state, where A and B
    have shape (embed_dim, rank) and (rank, embed_dim) respectively.
    Each loop iteration t uses its own DepthLoRA so behavior differs per step.
    """

    def __init__(self, embed_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting mechanism.

    Per-position scalar that learns when to stop the recurrent loop.
    Produces a halting probability p(t) at each step t; once cumulative
    probability exceeds *threshold*, the position stops receiving updates.

    Returns:
        weighted_state:  Σ_t  w(t) · h(t)  — the ACT-blended output
        act_loss:        ponder cost  (Σ cumul_prob)  for regularisation
    """

    def __init__(self, embed_dim: int, threshold: float = 0.99):
        super().__init__()
        self.halt_proj = nn.Linear(embed_dim, 1, bias=True)
        self.threshold = threshold

    def compute_halt_prob(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D) -> (B, T, 1) halt probability."""
        return torch.sigmoid(self.halt_proj(h))


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) comme utilisé dans LLaMA et autres LLMs modernes.
    
    RoPE encode les positions directement dans les queries et keys via des rotations,
    sans nécessiter de paramètres apprenables.
    
    Args:
        dim: Dimension de chaque tête d'attention (embed_dim // num_heads)
        max_seq_len: Longueur de séquence maximale
        base: Base pour le calcul des fréquences (10000 par défaut)
    """
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def rotate_half(self, x):
        """Rotation de moitié des dimensions."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, offset=0):
        """Applique RoPE aux queries et keys.
        
        Args:
            q: queries [batch, heads, seq_len, head_dim]
            k: keys [batch, heads, seq_len, head_dim]
            offset: position offset for KV cache (number of previously cached tokens)
        
        Returns:
            q_rot, k_rot: queries et keys avec positions encodées
        """
        seq_len = q.shape[2]
        
        cos = self.cos_cached[:, :, offset:offset + seq_len, :]
        sin = self.sin_cached[:, :, offset:offset + seq_len, :]
        
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Faster than LayerNorm: no mean-subtraction, no bias, single reduction.
    Used by LLaMA, Mistral, Gemma, Qwen.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function as described in the Super Tiny LM paper.
    SwiGLU(x) = (Swish(xW) ⊗ xV)W2
    where Swish(x) = SiLU(x) in PyTorch
    """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.v = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w(x)) * self.v(x))


class SelfAttention(nn.Module):
    """Multi-Head Attention with optional Grouped-Query Attention (GQA).

    When num_kv_heads < num_heads, K and V projections are smaller and each
    KV head is shared across num_heads // num_kv_heads query heads.
    When num_kv_heads == num_heads this is standard MHA (fully backward-compatible).
    """
    def __init__(self, embed_dim, heads, dropout, max_seq_len=2048, use_rope=True, num_kv_heads=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else heads
        self.head_dim = embed_dim // heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_rope = use_rope

        assert heads % self.num_kv_heads == 0, (
            f"num_heads ({heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )

        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RoPEEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        B, T, C = x.size()

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            q, k = self.rope(q, k, offset=offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # GQA: expand KV heads to match Q heads via repeat
        k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1) if self.num_kv_groups > 1 else k
        v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1) if self.num_kv_groups > 1 else v

        is_causal = past_kv is None
        attn = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=None,
            is_causal=is_causal,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out(attn)), new_kv

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1, hidden_dim=512, layerdrop=0.1,
                 shared_ff=None, max_seq_len=2048, use_rope=True, num_kv_heads=None):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads, dropout, max_seq_len=max_seq_len,
                                  use_rope=use_rope, num_kv_heads=num_kv_heads)
        self.ln1 = RMSNorm(embed_dim)
        self.ff = shared_ff if shared_ff is not None else SwiGLU(embed_dim, hidden_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layerdrop = layerdrop

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        if self.training and torch.rand(1).item() < self.layerdrop:
            return (x, None) if use_cache else x
        attn_out, new_kv = self.attn(self.ln1(x), mask, past_kv=past_kv, use_cache=use_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return (x, new_kv) if use_cache else x
    
    def forward_checkpointed(self, x, mask=None):
        """Version avec gradient checkpointing (training only, no cache)."""
        attn_out, _ = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class MiniGPT(PreTrainedModel):
    config_class = MiniGPTConfig
    
    def __init__(self, config=None, **kwargs):
        """
        Initialise le modèle MiniGPT.
        
        Args:
            config: Instance de MiniGPTConfig ou None. Si None, les paramètres
                   doivent être fournis via kwargs.
            **kwargs: Paramètres du modèle si config n'est pas fourni.
        """
        # Si config n'est pas fourni, créer une config à partir des kwargs
        if config is None:
            config = MiniGPTConfig(**kwargs)
        
        super().__init__(config)
        
        # Extraire les paramètres de la config
        vocab_size = config.vocab_size
        block_size = config.block_size
        embed_dim = config.embed_dim
        depth = config.depth
        heads = config.heads
        num_kv_heads = config.num_kv_heads
        dropout = config.dropout
        hidden_dim = config.hidden_dim
        weight_sharing = config.weight_sharing
        use_rope = config.use_rope
        use_gradient_checkpointing = config.use_gradient_checkpointing
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.use_rope = use_rope
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Positional embeddings uniquement si on n'utilise pas RoPE
        if not use_rope:
            self.pos_emb = nn.Embedding(block_size, embed_dim)
        else:
            self.pos_emb = None
        
        self.depth = depth
        self.weight_sharing = weight_sharing
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.hidden_dim = hidden_dim
        
        # Créer les blocs selon le type de weight sharing
        if weight_sharing == "none":
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, heads, dropout, hidden_dim, layerdrop=0.1,
                                 max_seq_len=block_size, use_rope=use_rope,
                                 num_kv_heads=num_kv_heads)
                for _ in range(depth)
            ])
        elif weight_sharing == "ffn":
            shared_ff = SwiGLU(embed_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, heads, dropout, hidden_dim, layerdrop=0.1,
                                 shared_ff=shared_ff, max_seq_len=block_size,
                                 use_rope=use_rope, num_kv_heads=num_kv_heads)
                for _ in range(depth)
            ])
        elif weight_sharing == "full":
            self.shared_block = TransformerBlock(embed_dim, heads, dropout, hidden_dim, layerdrop=0.1,
                                                 max_seq_len=block_size, use_rope=use_rope,
                                                 num_kv_heads=num_kv_heads)
            self.blocks = None
        else:
            raise ValueError(f"weight_sharing doit être 'none', 'ffn' ou 'full', pas '{weight_sharing}'")
        
        self.ln_f = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight #On réutilise les poids de la matrice token_emb pour les tetes 
        self.block_size = block_size
        self.apply(self._init_weights)

    def forward(self, idx):
        B, T = idx.shape
        
        # Token embeddings
        x = self.token_emb(idx)
        
        # Ajouter positional embeddings uniquement si on n'utilise pas RoPE
        if not self.use_rope:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        
        # Gradient checkpointing : économise VRAM en recalculant les activations
        if self.use_gradient_checkpointing and self.training:
            if self.weight_sharing == "full":
                for _ in range(self.depth):
                    x = checkpoint(self.shared_block.forward_checkpointed, x, mask, use_reentrant=False)
            else:
                for block in self.blocks:
                    x = checkpoint(block.forward_checkpointed, x, mask, use_reentrant=False)
        else:
            # Mode normal (pas de checkpointing)
            if self.weight_sharing == "full":
                for _ in range(self.depth):
                    x = self.shared_block(x, mask)
            else:
                for block in self.blocks:
                    x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
            
            
    def count_parameters(self):
        """Compte le nombre de paramètres selon le type de weight sharing et use_rope."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Détails par composant
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
            "head": 0,  # Head partage les poids avec embedding
            "weight_sharing": self.weight_sharing,
            "use_rope": self.use_rope
        }
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, min_new_tokens=0, eos_token_id=None):
        """
        Génération de texte avec contrôle de la diversité.
        
        Args:
            idx: Context initial [batch, seq_len]
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle la diversité (0.1=conservateur, 1.0=normal, 2.0=créatif)
            top_k: Garde seulement les k tokens les plus probables
            top_p: Nucleus sampling, garde les tokens dont la somme des probas = p
            min_new_tokens: Génère au moins ce nombre de tokens avant d'autoriser l'arrêt sur eos_token_id
            eos_token_id: Id du token EOS pour stopper la génération (optionnel)
        """
        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
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
