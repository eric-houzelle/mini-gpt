import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
        
        # Précalculer les fréquences
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Précalculer cos et sin pour toutes les positions
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def rotate_half(self, x):
        """Rotation de moitié des dimensions."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k):
        """Applique RoPE aux queries et keys.
        
        Args:
            q: queries [batch, heads, seq_len, head_dim]
            k: keys [batch, heads, seq_len, head_dim]
        
        Returns:
            q_rot, k_rot: queries et keys avec positions encodées
        """
        seq_len = q.shape[2]
        
        # Tronquer les embeddings si la séquence est plus courte
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Appliquer la rotation
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot


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
    def __init__(self, embed_dim, heads, dropout, max_seq_len=2048, use_rope=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.use_rope = use_rope
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = dropout 
        self.resid_dropout = nn.Dropout(dropout)
        
        # RoPE embeddings (pas de paramètres apprenables)
        if use_rope:
            self.rope = RoPEEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        # Appliquer RoPE aux queries et keys si activé
        if self.use_rope:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out(attn))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1, hidden_dim = 512, layerdrop=0.1, shared_ff=None, max_seq_len=2048, use_rope=True):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads, dropout, max_seq_len=max_seq_len, use_rope=use_rope)
        self.ln1 = nn.LayerNorm(embed_dim)
        # Utiliser un FFN partagé si fourni, sinon créer un nouveau
        self.ff = shared_ff if shared_ff is not None else SwiGLU(embed_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layerdrop = layerdrop

    def forward(self, x, mask=None):
        if self.training and torch.rand(1).item() < self.layerdrop:
            return x
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
    
    def forward_checkpointed(self, x, mask=None):
        """Version avec gradient checkpointing pour économiser VRAM."""
        return self.forward(x, mask)

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim=256, depth=8, heads=8, dropout = 0.1, hidden_dim = 512, weight_sharing="none", use_rope=True, use_gradient_checkpointing=False):
        """
        Args:
            weight_sharing: Type de partage de poids entre les blocs
                - "none": Pas de partage (comportement par défaut)
                - "ffn": Partage uniquement les FFN (comme MobiLlama)
                - "full": Partage FFN + Attention (comme ALBERT, réduction 90-95%)
            use_rope: Utiliser RoPE embeddings au lieu de positional embeddings appris
            use_gradient_checkpointing: Activer le gradient checkpointing pour économiser VRAM
                - True: Économise 50-70% de VRAM, +20-30% temps de calcul
                - False: VRAM normale, vitesse normale
        """
        super().__init__()
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
        
        # Créer les blocs selon le type de weight sharing
        if weight_sharing == "none":
            # Comportement original : chaque bloc a ses propres poids
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, heads, dropout, hidden_dim, layerdrop=0.1, 
                               max_seq_len=block_size, use_rope=use_rope) 
                for _ in range(depth)
            ])
        elif weight_sharing == "ffn":
            # Partage uniquement les FFN, attention séparée
            shared_ff = SwiGLU(embed_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, heads, dropout, hidden_dim, layerdrop=0.1, 
                               shared_ff=shared_ff, max_seq_len=block_size, use_rope=use_rope)
                for _ in range(depth)
            ])
        elif weight_sharing == "full":
            # ALBERT-style : un seul bloc réutilisé depth fois
            self.shared_block = TransformerBlock(embed_dim, heads, dropout, hidden_dim, layerdrop=0.1,
                                                max_seq_len=block_size, use_rope=use_rope)
            self.blocks = None  # On n'utilise pas de ModuleList dans ce cas
        else:
            raise ValueError(f"weight_sharing doit être 'none', 'ffn' ou 'full', pas '{weight_sharing}'")
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False) # on enleve bias pour que head et token_emb est la meme taille
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
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
            
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Génération de texte avec contrôle de la diversité.
        
        Args:
            idx: Context initial [batch, seq_len]
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle la diversité (0.1=conservateur, 1.0=normal, 2.0=créatif)
            top_k: Garde seulement les k tokens les plus probables
            top_p: Nucleus sampling, garde les tokens dont la somme des probas = p
        """
        for _ in range(max_new_tokens):
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
            idx = torch.cat((idx, next_token), dim=1)
        return idx
