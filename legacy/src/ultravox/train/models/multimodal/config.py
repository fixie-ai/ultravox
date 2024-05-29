import typing as t
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields

import peft
import transformers
from train.models.audio import encoders

LLM_NAME_MAP = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # ----------
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    # ----------
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "vicuna-13b": "lmsys/vicuna-13b-v1.5",
    # ----------
    "misral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
}


@dataclass(init=False)
class SpeechLMConfig(transformers.PretrainedConfig):
    llm_name: str = "tinyllama"
    audio_enc_name: str = "wav2vec2-base"
    audio_stride: int = 10
    use_cpu: bool = False
    audio_squeeze_type: str = "stack"
    init_type: str = "default"  # "default" or "small"
    # device_map: t.Union[str, t.Dict[str, int]] = "auto"

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        hf_names = set(kwargs.keys()) - names
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
        super().__init__(**{k: v for k, v in kwargs.items() if k in hf_names})
        self.__post_init__()

    def is_audio_enc_w2vbert(self):
        return "bert" in self.audio_enc_name

    def __post_init__(self):
        if self.audio_squeeze_type not in ["stride", "mean", "stack", "random"]:
            raise ValueError(f"Unknown audio_squeeze_type: {self.audio_squeeze_type}")

        # support shortened names
        if self.llm_name in LLM_NAME_MAP:
            self.llm_name = LLM_NAME_MAP[self.llm_name]

        if self.audio_enc_name in encoders.AUDIO_ENC_NAME_MAP:
            self.audio_enc_name = encoders.AUDIO_ENC_NAME_MAP[self.audio_enc_name]

        super().__init__()


@dataclass
class LoraConfigSimplified:
    """
    For some reason I get an error when I try to use the peft.LoraConfig class
    So this is a simplified proxy for that class
    """

    r: int = 0
    lora_alpha: float = 8
    target_modules: t.Optional[t.Union[t.List[str], str]] = field(
        default_factory=lambda: ["k_proj", "q_proj", "linear_k", "linear_q"]
    )


@dataclass
class FreezingConfig:
    llm_lora_config: LoraConfigSimplified = None
    audio_enc_lora_config: LoraConfigSimplified = None
    freeze_text_embeds: bool = False
    freeze_audio_embeds: bool = False

    def __post_init__(self):
        if self.llm_lora_config is not None:
            self.llm_lora_config = peft.LoraConfig(**asdict(self.llm_lora_config))

        if self.audio_enc_lora_config is not None:
            self.audio_enc_lora_config = peft.LoraConfig(
                **asdict(self.audio_enc_lora_config)
            )
