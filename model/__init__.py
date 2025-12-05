from .model import MiniGPT  # Pour compatibilité avec train.py et generate.py
from .modeling_minigpt_core import MiniGPTModel
from .modeling_minigpt import MiniGPTForCausalLM
from .configuration import MiniGPTConfig

# Enregistrer le modèle dans le mapping Hugging Face
try:
    from transformers import AutoModelForCausalLM, AutoModel
    AutoModelForCausalLM.register(MiniGPTConfig, MiniGPTForCausalLM)
    AutoModel.register(MiniGPTConfig, MiniGPTModel)
except Exception:
    # Si l'enregistrement échoue, ce n'est pas critique
    pass

__all__ = ["MiniGPT", "MiniGPTModel", "MiniGPTForCausalLM", "MiniGPTConfig"]

