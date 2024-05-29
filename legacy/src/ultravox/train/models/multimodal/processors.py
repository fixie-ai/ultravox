import typing as t
from dataclasses import dataclass

import transformers
from train.models import audio as audio_models

from .config import SpeechLMConfig


@dataclass
class SpeechLMProcessor(transformers.ProcessorMixin):
    tokenizer: transformers.LlamaTokenizer
    audio_processor: t.Union[
        transformers.Wav2Vec2Processor, transformers.Wav2Vec2BertProcessor
    ]
    total_audio_stride: int

    @staticmethod
    def from_config(config: SpeechLMConfig):
        audio_model_loader = audio_models.AudioEncLoader(config.audio_enc_name)
        total_audio_stride = audio_model_loader.audio_factor * config.audio_stride
        return SpeechLMProcessor(
            tokenizer=transformers.LlamaTokenizerFast.from_pretrained(config.llm_name),
            audio_processor=audio_model_loader.get_processor(),
            total_audio_stride=total_audio_stride,
        )
