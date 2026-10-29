from transformers import AutoConfig, AutoModelForCausalLM

from .model_wh import WhConfig, WhForCausalLM

# Make the local WH architecture discoverable by Transformers auto classes.
# Without this registration, AutoTokenizer falls back to the base
# PretrainedConfig when it inspects a checkpoint with model_type wh.
AutoConfig.register(WhConfig.model_type, WhConfig, exist_ok=True)
AutoModelForCausalLM.register(WhConfig, WhForCausalLM, exist_ok=True)

__all__ = ["WhConfig", "WhForCausalLM"]