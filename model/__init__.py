from .model import (
    MiniGPT,
    LTIInjection,
    DepthLoRA,
    ACTHalting,
)
from .modeling_minigpt_core import MiniGPTModel
from .modeling_minigpt import MiniGPTForCausalLM
from .configuration import MiniGPTConfig

# Enregistrer le modèle dans le mapping Hugging Face
try:
    from transformers import AutoModelForCausalLM, AutoModel
    AutoModelForCausalLM.register(MiniGPTConfig, MiniGPTForCausalLM)
    AutoModel.register(MiniGPTConfig, MiniGPTModel)
except Exception:
    pass

__all__ = [
    "MiniGPT",
    "MiniGPTModel",
    "MiniGPTForCausalLM",
    "MiniGPTConfig",
    "LTIInjection",
    "DepthLoRA",
    "ACTHalting",
]

