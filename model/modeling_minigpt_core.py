import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast

from .model import (
    RoPEEmbedding, RMSNorm, SwiGLU, SelfAttention, TransformerBlock,
    LTIInjection, DepthLoRA, ACTHalting,
)

from .configuration import MiniGPTConfig


class MiniGPTModel(nn.Module):
    """
    Modèle core MiniGPT — sans tête LM, pure architecture Transformer.
    NE DOIT PAS hériter de PreTrainedModel.

    Supporte 4 modes via weight_sharing:
        "none"              – couches indépendantes (classique)
        "ffn"               – FFN partagé entre couches
        "full"              – 1 bloc partagé appliqué depth fois
        "recurrent_depth"   – Recurrent-Depth Transformer (Prelude → boucle → Coda)
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

        if not config.use_rope:
            self.pos_emb = nn.Embedding(block_size, embed_dim)
        else:
            self.pos_emb = None

        def _make_block(**extra):
            return TransformerBlock(
                embed_dim, heads, dropout, hidden_dim,
                max_seq_len=block_size, use_rope=config.use_rope,
                num_kv_heads=num_kv_heads, **extra,
            )

        # ----- Recurrent-Depth Transformer -----
        if self.weight_sharing == "recurrent_depth":
            self.num_prelude = config.num_prelude_layers
            self.num_coda = config.num_coda_layers
            self.num_recurrent_steps = config.num_recurrent_steps

            self.prelude = nn.ModuleList([_make_block() for _ in range(self.num_prelude)])
            self.recurrent_block = _make_block()
            self.coda = nn.ModuleList([_make_block() for _ in range(self.num_coda)])

            self.use_lti = config.use_lti_injection
            if self.use_lti:
                self.lti = LTIInjection(embed_dim)

            self.depth_lora_rank = config.depth_lora_rank
            if self.depth_lora_rank > 0:
                self.depth_loras = nn.ModuleList([
                    DepthLoRA(embed_dim, self.depth_lora_rank)
                    for _ in range(self.num_recurrent_steps)
                ])

            self.use_act = config.use_act_halting
            if self.use_act:
                self.act = ACTHalting(embed_dim, threshold=config.act_halt_threshold)

            self.blocks = None

        # ----- Standard modes -----
        elif self.weight_sharing == "none":
            self.blocks = nn.ModuleList([_make_block() for _ in range(depth)])

        elif self.weight_sharing == "ffn":
            shared_ff = SwiGLU(embed_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                _make_block(shared_ff=shared_ff) for _ in range(depth)
            ])

        elif self.weight_sharing == "full":
            self.shared_block = _make_block()
            self.blocks = None

        else:
            raise ValueError(
                f"weight_sharing must be 'none', 'ffn', 'full' or 'recurrent_depth', "
                f"got '{self.weight_sharing}'"
            )

        self.ln_f = RMSNorm(embed_dim)
        self._act_loss = torch.tensor(0.0)
        self._inference_recurrent_steps = None

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def get_output_embeddings(self):
        return None

    @property
    def act_loss(self) -> torch.Tensor:
        """Ponder cost from ACT — add to training loss as regularisation."""
        return self._act_loss

    def set_inference_recurrent_steps(self, steps: int | None):
        """Override the number of recurrent steps used at inference time.

        Allows deeper reasoning without retraining. LoRA adapters are
        reused cyclically for steps beyond the trained count.
        Set to None to revert to the training default.
        """
        self._inference_recurrent_steps = steps

    @property
    def effective_recurrent_steps(self) -> int:
        """Number of recurrent steps for the current forward pass."""
        if self.weight_sharing != "recurrent_depth":
            return 0
        if not self.training and self._inference_recurrent_steps is not None:
            return self._inference_recurrent_steps
        return self.num_recurrent_steps

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

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

        if self.weight_sharing == "recurrent_depth":
            x = self._forward_recurrent_depth(
                x, mask, past_key_values, use_cache, new_key_values,
            )
        elif self.use_gradient_checkpointing and self.training:
            x = self._forward_checkpointed(x, mask)
        elif use_cache:
            x = self._forward_cached(x, mask, past_key_values, new_key_values)
        else:
            x = self._forward_standard(x, mask)

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

    # ------------------------------------------------------------------ #
    #  RDT forward path                                                   #
    # ------------------------------------------------------------------ #

    def _forward_recurrent_depth(self, x, mask, past_key_values, use_cache, new_key_values):
        """Prelude -> Recurrent loop (LTI + ACT + LoRA) -> Coda."""
        kv_idx = 0

        # --- Prelude ---
        for block in self.prelude:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block.forward_checkpointed, x, mask, use_reentrant=False)
            elif use_cache:
                past_kv = past_key_values[kv_idx] if past_key_values is not None else None
                x, kv = block(x, mask, past_kv=past_kv, use_cache=True)
                new_key_values.append(kv)
                kv_idx += 1
            else:
                x = block(x, mask)

        encoded_input = x  # snapshot for LTI re-injection

        # --- Recurrent loop ---
        n_steps = self.effective_recurrent_steps
        if self.use_act and not self.training:
            x = self._recurrent_loop_act_inference(
                x, encoded_input, mask, past_key_values, use_cache, new_key_values, kv_idx,
                n_steps,
            )
            kv_idx += n_steps
        else:
            x, kv_idx = self._recurrent_loop(
                x, encoded_input, mask, past_key_values, use_cache, new_key_values, kv_idx,
                n_steps,
            )

        # --- Coda ---
        for block in self.coda:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block.forward_checkpointed, x, mask, use_reentrant=False)
            elif use_cache:
                past_kv = past_key_values[kv_idx] if past_key_values is not None else None
                x, kv = block(x, mask, past_kv=past_kv, use_cache=True)
                new_key_values.append(kv)
                kv_idx += 1
            else:
                x = block(x, mask)

        return x

    def _recurrent_loop(self, x, encoded_input, mask, past_key_values, use_cache, new_key_values, kv_idx,
                         n_steps=None):
        """Standard recurrent loop (training, or eval without ACT).

        During training with ACT enabled, we compute halt probabilities and
        accumulate the weighted output + ponder cost for the loss.
        """
        if n_steps is None:
            n_steps = self.num_recurrent_steps
        B, T, D = x.shape
        device = x.device

        use_act_train = self.use_act and self.training
        if use_act_train:
            cumul_prob = torch.zeros(B, T, 1, device=device)
            remainder = torch.ones(B, T, 1, device=device)
            weighted_state = torch.zeros_like(x)
            still_running = torch.ones(B, T, 1, device=device, dtype=torch.bool)

        for t in range(n_steps):
            if self.use_gradient_checkpointing and self.training:
                block_out = checkpoint(
                    self.recurrent_block.forward_checkpointed, x, mask, use_reentrant=False,
                )
            elif use_cache:
                past_kv = past_key_values[kv_idx] if past_key_values is not None else None
                block_out, kv = self.recurrent_block(x, mask, past_kv=past_kv, use_cache=True)
                new_key_values.append(kv)
                kv_idx += 1
            else:
                block_out = self.recurrent_block(x, mask)

            # LTI injection: h(t+1) = A·h(t) + B·e + block(h(t))
            if self.use_lti:
                x = self.lti(x, encoded_input) + block_out
            else:
                x = block_out

            if self.depth_lora_rank > 0:
                lora_idx = t % len(self.depth_loras)
                x = x + self.depth_loras[lora_idx](x)

            # ACT accumulation (training only)
            if use_act_train:
                p = self.act.compute_halt_prob(x)
                still_running_f = still_running.float()
                new_halted = (cumul_prob + p > self.act.threshold) & still_running
                still_running = still_running & ~new_halted

                w = torch.where(new_halted, remainder, p * still_running_f)
                weighted_state = weighted_state + w * x
                cumul_prob = cumul_prob + p * still_running_f
                remainder = remainder - p * still_running_f

        if use_act_train:
            remainder_mask = still_running.float()
            weighted_state = weighted_state + remainder * remainder_mask * x
            self._act_loss = cumul_prob.sum() / (B * T)
            x = weighted_state

        return x, kv_idx

    def _recurrent_loop_act_inference(
        self, x, encoded_input, mask, past_key_values, use_cache, new_key_values, kv_idx,
        n_steps=None,
    ):
        """Recurrent loop with early exit per position at inference time."""
        if n_steps is None:
            n_steps = self.num_recurrent_steps
        B, T, D = x.shape
        device = x.device

        cumul_prob = torch.zeros(B, T, 1, device=device)
        remainder = torch.ones(B, T, 1, device=device)
        weighted_state = torch.zeros_like(x)
        still_running = torch.ones(B, T, 1, device=device, dtype=torch.bool)

        for t in range(n_steps):
            if not still_running.any():
                if use_cache:
                    new_key_values.append(None)
                continue

            if use_cache:
                past_kv = past_key_values[kv_idx + t] if past_key_values is not None else None
                block_out, kv = self.recurrent_block(x, mask, past_kv=past_kv, use_cache=True)
                new_key_values.append(kv)
            else:
                block_out = self.recurrent_block(x, mask)

            if self.use_lti:
                x = self.lti(x, encoded_input) + block_out
            else:
                x = block_out

            if self.depth_lora_rank > 0:
                lora_idx = t % len(self.depth_loras)
                x = x + self.depth_loras[lora_idx](x)

            p = self.act.compute_halt_prob(x)
            still_running_f = still_running.float()
            new_halted = (cumul_prob + p > self.act.threshold) & still_running
            still_running = still_running & ~new_halted

            w = torch.where(new_halted, remainder, p * still_running_f)
            weighted_state = weighted_state + w * x
            cumul_prob = cumul_prob + p * still_running_f
            remainder = remainder - p * still_running_f

        remainder_mask = still_running.float()
        weighted_state = weighted_state + remainder * remainder_mask * x
        self._act_loss = torch.tensor(0.0, device=device)

        return weighted_state

    # ------------------------------------------------------------------ #
    #  Classic forward helpers (unchanged logic, just extracted)           #
    # ------------------------------------------------------------------ #

    def _forward_standard(self, x, mask):
        if self.weight_sharing == "full":
            for _ in range(self.depth):
                x = self.shared_block(x, mask)
        else:
            for block in self.blocks:
                x = block(x, mask)
        return x

    def _forward_cached(self, x, mask, past_key_values, new_key_values):
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
        return x

    def _forward_checkpointed(self, x, mask):
        if self.weight_sharing == "full":
            for _ in range(self.depth):
                x = checkpoint(self.shared_block.forward_checkpointed, x, mask, use_reentrant=False)
        else:
            for block in self.blocks:
                x = checkpoint(block.forward_checkpointed, x, mask, use_reentrant=False)
        return x

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    def count_parameters(self):
        """Compte le nombre de paramètres selon le mode."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        token_emb_params = self.token_emb.weight.numel()
        pos_emb_params = self.pos_emb.weight.numel() if self.pos_emb is not None else 0
        embedding_params = token_emb_params + pos_emb_params

        rdt_extra = 0
        if self.weight_sharing == "recurrent_depth":
            block_params = (
                sum(p.numel() for p in self.prelude.parameters())
                + sum(p.numel() for p in self.recurrent_block.parameters())
                + sum(p.numel() for p in self.coda.parameters())
            )
            if self.use_lti:
                rdt_extra += sum(p.numel() for p in self.lti.parameters())
            if self.depth_lora_rank > 0:
                rdt_extra += sum(p.numel() for p in self.depth_loras.parameters())
            if self.use_act:
                rdt_extra += sum(p.numel() for p in self.act.parameters())
        elif self.weight_sharing == "full":
            block_params = sum(p.numel() for p in self.shared_block.parameters())
        else:
            block_params = sum(p.numel() for p in self.blocks.parameters())

        result = {
            "total": total,
            "trainable": trainable,
            "embedding": embedding_params,
            "token_emb": token_emb_params,
            "pos_emb": pos_emb_params,
            "blocks": block_params,
            "head": 0,
            "weight_sharing": self.weight_sharing,
            "use_rope": self.use_rope,
        }
        if self.weight_sharing == "recurrent_depth":
            result["rdt_extra"] = rdt_extra
        return result
