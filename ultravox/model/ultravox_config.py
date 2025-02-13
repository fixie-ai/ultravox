import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional

import transformers


@dataclasses.dataclass
class LoraConfigSimplified:
    """
    Low Rank Approximation (LoRA) configuration.

    Used for language and audio models separately.
    """

    # The rank of the approximation
    r: int = 0
    lora_alpha: float = 8
    target_modules: Optional[List[str]] = dataclasses.field(
        default_factory=lambda: ["k_proj", "q_proj", "linear_k", "linear_q"]
    )
    # A list of module names regex patterns to unfreeze. Only used if r == 0.
    unfreeze_layers: Optional[List[str]] = None


class LossFunction(str, Enum):
    CrossEntropy = "ce"
    KL_Divergence = "kl"


@dataclasses.dataclass
class LossConfig:
    loss_function: LossFunction = LossFunction.CrossEntropy
    kl_temperature: float = 2.0

    @property
    def requires_alt_fields(self):
        return self.loss_function == LossFunction.KL_Divergence


class UltravoxConfig(transformers.PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UltravoxForConditionalGeneration`]. It is used to instantiate an
    Ultravox model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`WhisperConfig`,  *optional*):
            Custom audio config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        audio_token_index (`int`, *optional*, defaults to 32000):
            The audio token index to encode the audio prompt.
        stack_factor (`int`, *optional*, defaults to 8):
            Audio downsampling factor for the multimodal projector.
        norm_init (`float`, *optional*, defaults to 0.4):
            The initialization value for the layer normalization.
        projector_act (`str`, *optional*, defaults to `"swiglu"`):
            The activation function used by the multimodal projector.
        text_model_lora_config (`LoraConfigSimplified`, *optional*):
            The LoRA configuration for finetuning the text model.
        audio_model_lora_config (`LoraConfigSimplified`, *optional*):
            The LoRA configuration for finetuning the audio model.
        audio_latency_block_size (`int`, *optional*, defaults to `None`):
            The latency block size for simulating audio streaming.


    Example:

    ```python
    >>> from transformers import UltravoxModel, WhisperConfig, UltravoxConfig, LlamaConfig

    >>> # Initializing an audio encoder config
    >>> audio_config = WhisperConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a default configuration
    >>> configuration = UltravoxConfig(audio_config, text_config)

    >>> # Initializing a completely untrained model from the configuration
    >>> model = UltravoxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initialize a model from pretrained checkpoints and random projector weights
    >>> config = UltravoxConfig(audio_model_id="openai/whisper-tiny", text_model_id="meta-llama/Llama-2-7b-chat-hf")
    ```"""

    model_type = "ultravox"
    is_composition = False

    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        audio_model_id: Optional[str] = None,
        text_model_id: Optional[str] = None,
        ignore_index: int = -100,
        hidden_size: int = 4096,
        stack_factor: int = 8,
        norm_init: float = 0.4,
        projector_act: str = "swiglu",
        projector_ln_mid: bool = False,  # defaults to False for compatibility with v0.4.1 and below
        text_model_lora_config: Optional[LoraConfigSimplified] = None,
        audio_model_lora_config: Optional[LoraConfigSimplified] = None,
        audio_latency_block_size: Optional[int] = None,
        **kwargs,
    ):
        self.ignore_index = ignore_index

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id

        self.hidden_size = hidden_size
        self.stack_factor = stack_factor
        self.norm_init = norm_init
        self.projector_act = projector_act
        self.projector_ln_mid = projector_ln_mid
        if text_model_id is not None:
            self.text_config: transformers.LlamaConfig = (
                transformers.AutoConfig.from_pretrained(text_model_id)
            )
        else:
            text_config = text_config or {}
            self.text_config = transformers.CONFIG_MAPPING[
                text_config.get("model_type", "llama")
            ](**text_config)

        if audio_model_id is not None:
            self.audio_config: transformers.PretrainedConfig = (
                transformers.AutoConfig.from_pretrained(audio_model_id)
            )
        else:
            audio_config = audio_config or {}
            self.audio_config = transformers.CONFIG_MAPPING[
                audio_config.get("model_type", "whisper")
            ](**audio_config)

        self.text_model_lora_config = (
            text_model_lora_config
            if isinstance(text_model_lora_config, dict)
            else dataclasses.asdict(text_model_lora_config or LoraConfigSimplified())
        )
        self.audio_model_lora_config = (
            audio_model_lora_config
            if isinstance(audio_model_lora_config, dict)
            else dataclasses.asdict(audio_model_lora_config or LoraConfigSimplified())
        )
        self.audio_latency_block_size = audio_latency_block_size

        self.vocab_size = self.text_config.vocab_size

        self.initializer_range = self.text_config.initializer_range

        super().__init__(**kwargs)

    def to_diff_dict(self) -> Dict[str, Any]:
        diff_dict = super().to_diff_dict()

        # remove text_config and audio_config if text_model_id and audio_model_id are present
        if self.text_model_id is not None:
            diff_dict.pop("text_config", None)
        elif "text_config" in diff_dict:
            diff_dict["text_config"].pop("_attn_implementation_autoset", None)

        if self.audio_model_id is not None:
            diff_dict.pop("audio_config", None)
        elif "audio_config" in diff_dict:
            diff_dict["audio_config"].pop("_attn_implementation_autoset", None)

        return diff_dict
