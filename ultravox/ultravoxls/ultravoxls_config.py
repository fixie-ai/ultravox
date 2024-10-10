import dataclasses
from typing import Any, Dict, Optional

import transformers

from ultravox.model.ultravox_config import LoraConfigSimplified


class UltravoxLSConfig(transformers.PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UltravoxForConditionalGeneration`]. It is used to instantiate an
    UltravoxLS model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Must be `LlamaConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        norm_init (`float`, *optional*, defaults to 0.4):
            The initialization value for the layer normalization.
        pad_to_multiple_of (`int`, *optional*, defaults to 8):
            Pads all input_ids to a multiple of the provided value. Some CUDA devices benefit from this.
            See https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/ for more information.
            Usually this is done in the data preprocessing step, but it has been pushed to model since wavtokenizer is part of model.
        text_model_lora_config (`LoraConfigSimplified`, *optional*):
            The LoRA configuration for finetuning the text model.

    Example:

    ```python
    >>> from transformers import UltravoxForConditionalGeneration, Wav2Vec2Config, UltravoxLSConfig, LlamaConfig

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a default configuration
    >>> configuration = UltravoxLSConfig(text_config)

    >>> # Initializing a completely untrained model from the configuration
    >>> model = UltravoxForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initialize a model from pretrained checkpoints and random projector weights
    >>> config = UltravoxLSConfig(text_model_id="meta-llama/Llama-2-7b-chat-hf")
    ```"""

    model_type = "ultravoxlsm"
    is_composition = False

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        text_model_id: Optional[str] = None,
        vocab_size: int = 4096,
        ignore_index: int = -100,
        hidden_size: int = 4096,
        norm_init: float = 0.4,
        pad_to_multiple_of: int = 8,
        text_model_lora_config: Optional[LoraConfigSimplified] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size

        self.ignore_index = ignore_index

        self.text_model_id = text_model_id

        self.hidden_size = hidden_size
        self.norm_init = norm_init
        self.pad_to_multiple_of = pad_to_multiple_of

        if text_model_id is not None:
            self.text_config: transformers.LlamaConfig = (
                transformers.AutoConfig.from_pretrained(text_model_id)
            )
        else:
            text_config = text_config or {}
            self.text_config = transformers.CONFIG_MAPPING[
                text_config.get("model_type", "llama")
            ](**text_config)

        self.text_model_lora_config = (
            text_model_lora_config
            if isinstance(text_model_lora_config, dict)
            else dataclasses.asdict(text_model_lora_config or LoraConfigSimplified())
        )

        self.initializer_range = self.text_config.initializer_range

        super().__init__(**kwargs)

    def to_diff_dict(self) -> Dict[str, Any]:
        diff_dict = super().to_diff_dict()

        # remove text_config if text_model_id
        if self.text_model_id is not None:
            diff_dict.pop("text_config", None)

        return diff_dict
