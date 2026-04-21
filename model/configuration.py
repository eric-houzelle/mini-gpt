from transformers import PretrainedConfig


class MiniGPTConfig(PretrainedConfig):
    """
    Configuration pour le modèle MiniGPT.
    
    Cette classe hérite de PretrainedConfig pour être compatible avec
    l'écosystème Hugging Face.

    Recurrent-Depth Transformer (RDT) — activé via weight_sharing="recurrent_depth":
        Architecture Prelude → Recurrent Block (boucle T fois) → Coda.
        Inspiré de OpenMythos / Parcae (Prairie et al., 2026).

        Paramètres RDT:
            num_prelude_layers: couches uniques avant la boucle (encodage initial)
            num_coda_layers: couches uniques après la boucle (décodage final)
            num_recurrent_steps: nombre d'itérations T du bloc récurrent
            use_lti_injection: active l'injection LTI h(t+1) = A·h(t) + B·e + block(h(t))
            use_act_halting: active Adaptive Computation Time (arrêt dynamique par position)
            act_halt_threshold: seuil de probabilité de halting pour ACT
            depth_lora_rank: rang des adaptateurs LoRA par profondeur (0 = désactivé)
    """
    model_type = "minigpt"
    
    def __init__(
        self,
        vocab_size=32000,
        block_size=256,
        embed_dim=256,
        depth=8,
        heads=8,
        num_kv_heads=None,
        dropout=0.1,
        hidden_dim=512,
        weight_sharing="none",
        use_rope=True,
        use_gradient_checkpointing=False,
        # Recurrent-Depth Transformer params
        num_prelude_layers=2,
        num_coda_layers=2,
        num_recurrent_steps=8,
        use_lti_injection=True,
        use_act_halting=True,
        act_halt_threshold=0.99,
        depth_lora_rank=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.weight_sharing = weight_sharing
        self.use_rope = use_rope
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # RDT
        self.num_prelude_layers = num_prelude_layers
        self.num_coda_layers = num_coda_layers
        self.num_recurrent_steps = num_recurrent_steps
        self.use_lti_injection = use_lti_injection
        self.use_act_halting = use_act_halting
        self.act_halt_threshold = act_halt_threshold
        self.depth_lora_rank = depth_lora_rank

