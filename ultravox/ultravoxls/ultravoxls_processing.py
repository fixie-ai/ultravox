from typing import Optional, Union

import huggingface_hub
import librosa
import numpy as np
import torch
import transformers

from third_party.tokenizer import wav_tokenizer
from ultravox.data import datasets

SAMPLE_RATE_LS = 24000
HF_MODEL_NAME = "novateur/WavTokenizer"
CHECKPOINT_FILE_NAME = "WavTokenizer_small_600_24k_4096.ckpt"
CONFIG_FILE_NAME = (
    "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
)


class UltravoxLSProcessor:
    tokenizer: transformers.PreTrainedTokenizerBase

    def __init__(
        self,
        model_device: str,
    ):
        self.model_device = model_device
        # Create tokenizer
        config_path = "../../third_party/tokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        model_path = huggingface_hub.hf_hub_download(
            repo_id=HF_MODEL_NAME, filename=CHECKPOINT_FILE_NAME
        )
        self.tokenizer = wav_tokenizer.CustomWavTokenizer(config_path, model_path)

    def dataproc(self, sample: datasets.VoiceSample):
        if sample.audio is not None:
            audio = sample.audio
            sample_rate = sample.sample_rate
            # Normalize audio to float32.
            if audio.dtype == np.int16:
                audio = audio / np.float32(32768.0)
            if audio.dtype not in [np.float64, np.float32]:
                raise ValueError("Audio must be float64 or float32 or int16")

            # Convert to tensor, resampling to 24kHz if needed.
            if sample_rate != SAMPLE_RATE_LS:
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE_LS
                )
            audio_input = torch.from_numpy(audio)
            if audio_input.ndim == 1:
                audio_input = audio_input.unsqueeze(0)

        else:
            raise ValueError("Audio input is required for ultravoxls inference")

        inputs = self.__call__(
            audio=audio_input, return_tensors="pt", sampling_rate=SAMPLE_RATE_LS
        )
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
        return inputs

    def decode(self, output_tokens: torch.Tensor, skip_special_tokens: bool = True):
        return self.tokenizer.decode(
            output_tokens, skip_special_tokens=skip_special_tokens
        )

    def __call__(
        self,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_tensors: Optional[
            Union[str, transformers.TensorType]
        ] = transformers.TensorType.PYTORCH,
        **kwargs,
    ) -> transformers.BatchFeature:
        tokenized_audio = self.tokenizer.encode(audio)
        if len(tokenized_audio.shape) == 3:
            tokenized_audio = tokenized_audio.squeeze(1)

        attention_mask = np.ones_like(tokenized_audio, dtype=np.int64)
        data = {"input_ids": tokenized_audio, "attention_mask": attention_mask}
        return transformers.BatchFeature(data=data, tensor_type=return_tensors)
