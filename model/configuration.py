from transformers import PretrainedConfig


class MiniGPTConfig(PretrainedConfig):
    """
    Configuration pour le modèle MiniGPT.
    
    Cette classe hérite de PretrainedConfig pour être compatible avec
    l'écosystème Hugging Face.
    """
    model_type = "minigpt"
    
    def __init__(
        self,
        vocab_size=32000,
        block_size=256,
        embed_dim=256,
        depth=8,
        heads=8,
        dropout=0.1,
        hidden_dim=512,
        weight_sharing="none",
        use_rope=True,
        use_gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.weight_sharing = weight_sharing
        self.use_rope = use_rope
        self.use_gradient_checkpointing = use_gradient_checkpointing

