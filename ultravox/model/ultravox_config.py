import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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

class LossFunction(str, Enum):
    Response_CE = "Response_CE"
    Response_KL = "Response_KL"
    Input_KL = "Input_KL"
    CIF_L1 = "CIF_L1"

class AdapterType(str, Enum):
    STACKING = "STACKING"
    CFORMER = "CFORMER"

@dataclasses.dataclass
class LossConfig:
    loss_weights: Dict[LossFunction, float] = dataclasses.field(default_factory=lambda: {LossFunction.Response_KL: 1.0})    
    kl_temperature: float = 2.0

    @property
    def requires_alt_fields(self):
        return any(lf in self.loss_weights for lf in [LossFunction.Input_KL, LossFunction.Response_KL])
    
    def add_adapter_losses(self, adapter_type: AdapterType):
        if adapter_type == AdapterType.CFORMER and LossFunction.CIF_L1 not in self.loss_weights:
            self.loss_weights[LossFunction.CIF_L1] = 1.0

    @property
    def contains_kl_loss(self):
        return any(lf in self.loss_weights for lf in [LossFunction.Input_KL, LossFunction.Response_KL])

@dataclasses.dataclass
class UltravoxCFormerAdapterConfig:
    """
    CFormer Adapter configuration.

    CIF+Transformer-based adapter to segment speech into continuous speech tokens with 1:1 correspondence to text tokens.
"""
    num_pre_cif_layers: int = 2
    num_post_cif_layers: int = 2


@dataclasses.dataclass
class UltravoxStackingAdapterConfig:
    """
    Stacking Adapter configuration.

    Stacking+Convolutions-based adapter to segment speech into continuous speech tokens at a fixed downsampling rate.
"""
    stack_factor: int = 8
    activation: str = "swiglu"

ADAPTER_CONFIG_MAP: Dict[AdapterType, Any] = {
    AdapterType.STACKING: UltravoxCFormerAdapterConfig,
    AdapterType.CFORMER: UltravoxStackingAdapterConfig,
}

class UltravoxConfig(transformers.PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UltravoxForConditionalGeneration`]. It is used to instantiate an
    Ultravox model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Wav2Vec2Config`,  *optional*):
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

    Example:

    ```python
    >>> from transformers import UltravoxForConditionalGeneration, Wav2Vec2Config, UltravoxConfig, LlamaConfig

    >>> # Initializing an audio encoder config
    >>> audio_config = Wav2Vec2Config()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a default configuration
    >>> configuration = UltravoxConfig(audio_config, text_config)

    >>> # Initializing a completely untrained model from the configuration
    >>> model = UltravoxForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initialize a model from pretrained checkpoints and random projector weights
    >>> config = UltravoxConfig(audio_model_id="facebook/wav2vec2-base-960h", text_model_id="meta-llama/Llama-2-7b-chat-hf")
    ```"""

    model_type = "ultravox"
    is_composition = False

    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        adapter_config: Union[UltravoxStackingAdapterConfig, UltravoxCFormerAdapterConfig] = None,
        audio_model_id: Optional[str] = None,
        text_model_id: Optional[str] = None,
        adapter_type: AdapterType = AdapterType.STACKING,
        ignore_index: int = -100,
        audio_token_index: int = 32000,
        hidden_size: int = 4096,
        stack_factor: int = 8,
        norm_init: float = 0.4,
        projector_act: str = "swiglu",
        text_model_lora_config: Optional[LoraConfigSimplified] = None,
        audio_model_lora_config: Optional[LoraConfigSimplified] = None,
        **kwargs,
    ):
        self.ignore_index = ignore_index

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.audio_token_index = audio_token_index

        self.hidden_size = hidden_size
        self.stack_factor = stack_factor
        self.norm_init = norm_init
        self.projector_act = projector_act

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
                audio_config.get("model_type", "wav2vec2")
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

        self.adapter_type = adapter_type
        self.adapter_config = dataclasses.asdict(adapter_config or ADAPTER_CONFIG_MAP[adapter_type]())

        self.vocab_size = self.text_config.vocab_size

        self.initializer_range = self.text_config.initializer_range

        super().__init__(**kwargs)

UltravoxConfig.register_for_auto_class()
transformers.AutoConfig.register("ultravox", UltravoxConfig)
