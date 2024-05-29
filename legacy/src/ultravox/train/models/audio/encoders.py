import enum
import logging
import os
import typing as t

import torch
import transformers
import transformers.models.wav2vec2_bert.modeling_wav2vec2_bert as w2v2bert
import transformers.models.whisper.modeling_whisper as whisper

AUDIO_ENC_NAME_MAP = {
    "wav2vec2": "facebook/wav2vec2-base-960h",
    # ----------
    # TODO: use this or `ylacombe/w2v-bert-2.0`?
    # read comment in encoders.py
    "wav2vec2-bert": "hf-audio/wav2vec2-bert-CV16-en",
    "wav2vec2-bert-enconly": "hf-audio/wav2vec2-bert-CV16-en",
    # ----- HuBERT -----
    "hubert-base": "facebook/hubert-base-ls960",
    "hubert-large": "facebook/hubert-large-ls960-ft",
    "hubert-xlarge": "facebook/hubert-xlarge-ls960-ft",
    # ----- Whisper -----
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-medium": "openai/whisper-medium",
}


class AudioEncType(enum.Enum):
    Wav2Vec2 = "wav2vec2"
    Wav2Vec2Bert = "wav2vec2-bert"
    # Wav2Vec2BertEncOnly = "wav2vec2-bert-enconly"
    HuBERT = "hubert"
    Whisper = "whisper"

    @staticmethod
    def from_str(s: str) -> "AudioEncType":
        if "whisper" in s:
            return AudioEncType.Whisper
        if "hubert" in s:
            return AudioEncType.HuBERT
        if "wav2vec2-bert" in s:
            return AudioEncType.Wav2Vec2Bert
        if "wav2vec2" in s:
            return AudioEncType.Wav2Vec2
        raise ValueError(f"Unknown model_name: {s}")


class AudioEncLoader:
    model_class = transformers.AutoModel
    audio_factor: int = 320
    """
    Describes the ratio: len(waveform) / len(audio_feates)
    Given a fixed sampling rate of 16kHz, the frequency of the audio features is:
    16kHz / audio_factor, which is 50Hz in all cases by the full w2v2bert model.
    """

    def __init__(self, model_name: str, dtype: torch.dtype = torch.float32):
        self.model_name = model_name
        self.processor_name = model_name
        self.dtype = dtype

        model_type = AudioEncType.from_str(model_name)

        if model_type in (AudioEncType.Wav2Vec2, AudioEncType.HuBERT):
            # All HuBERT models use the same processor as Wav2Vec2
            # The HuBERT-base processor is broken in HF, so I'm forcefully replacing the name for all
            self.processor_name = "facebook/wav2vec2-base-960h"

        if model_type != AudioEncType.Wav2Vec2Bert:
            if self.dtype != torch.float32:
                logging.warn(
                    f"Wav2Vec2, HuBERT, and Whisper do not work with Half precision dtypes. You chose {dtype}. Forcing Float32."
                )
                self.dtype = torch.float32

        if model_type == AudioEncType.Wav2Vec2Bert:
            raise ValueError("Wav2Vec2Bert is not supported at the moment.")

        # TODO: this doesn't actually work, as a result wav2vec2-bert is broken
        # if "bert-enconly" in model_name:
        #     # The encoder only model (without the adapter head)
        #     self.model_class = w2v2bert.Wav2Vec2BertEncoder
        #     assert False
        # elif "wav2vec2-bert" in model_name:
        #     # The adapter reduces number of frames by 2
        #     self.audio_factor *= 2
        #     assert False

    def get_model(
        self,
    ) -> t.Union[
        transformers.Wav2Vec2Model,
        # transformers.Wav2Vec2BertModel,
        transformers.HubertModel,
        whisper.WhisperEncoder,
        w2v2bert.Wav2Vec2BertEncoder,
    ]:
        model = self.model_class.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            token=os.environ.get("HF_ACCESS_TOKEN", None),
        )

        if isinstance(model, transformers.WhisperForConditionalGeneration):
            # WhisperForConditionalGeneration -> WhisperModel
            model = model.model
        if isinstance(model, (w2v2bert.Wav2Vec2BertModel, whisper.WhisperModel)):
            # Wav2Vec2BertModel -> Wav2Vec2BertEncoder
            # WhisperModel -> WhisperEncoder
            model = model.encoder

        assert isinstance(
            model,
            (
                transformers.Wav2Vec2Model,
                transformers.HubertModel,
                w2v2bert.Wav2Vec2BertEncoder,
                whisper.WhisperEncoder,
            ),
        )
        return model

    def get_processor(
        self,
    ) -> t.Union[
        transformers.Wav2Vec2Processor,
        transformers.Wav2Vec2BertProcessor,
        transformers.WhisperProcessor,
    ]:
        return transformers.AutoProcessor.from_pretrained(self.processor_name)


# TODO: use which model_name?
# processor = transformers.AutoProcessor.from_pretrained(
#     "hf-audio/wav2vec2-bert-CV16-en"
# )
# # I checked and this is the same as `facebook/w2v-bert-2.0`
# # The only difference is that the processor for this one works
# model = transformers.Wav2Vec2BertModel.from_pretrained("ylacombe/w2v-bert-2.0")

# W2V-BERT-2.0 token rate is
# - 50Hz (20ms) after processor is applied and up to last conv layer (extract_features)
# - 25Hz (40ms) after model (last_hidden_state)
