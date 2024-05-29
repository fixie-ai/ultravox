import typing as t

import transformers

from . import audio as audio_models
from . import multimodal as multimodal_models
from . import text as text_models


def create_audiollm_model(
    config: multimodal_models.SpeechLMConfig,
) -> t.Tuple[
    multimodal_models.SpeechLM,
    transformers.LlamaTokenizer,
    transformers.Wav2Vec2Processor,
    int,
]:
    audio_enc, audio_proc, audio_factor = audio_models.get_audio_encoder(
        config.audio_enc_name, dtype=config.dtype
    )
    llm, tokenizer = text_models.get_llm(
        model_name=config.llm_name, cpu=config.use_cpu, dtype=config.dtype
    )

    model = multimodal_models.SpeechLM(
        audio_enc=audio_enc,
        llm=llm,
        audio_stride=config.audio_stride,
        dtype=config.dtype,
    )

    # TODO: model to device

    return model, tokenizer, audio_proc, audio_factor * config.audio_stride
