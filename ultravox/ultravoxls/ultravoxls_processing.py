import dataclasses
from typing import Optional, Union

import librosa
import numpy as np
import torch
import transformers

from ultravox.data import data_sample

SAMPLE_RATE_LS = 24000


class UltravoxLSProcessor:
    def dataproc(self, sample: data_sample.VoiceSample):
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
        return inputs

    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        return_tensors: Optional[
            Union[str, transformers.TensorType]
        ] = transformers.TensorType.PYTORCH,
        **kwargs,
    ) -> transformers.BatchFeature:
        data = {"audio": audio}
        return transformers.BatchFeature(data=data, tensor_type=return_tensors)


@dataclasses.dataclass
class DataCollatorForLSM:
    def __call__(self, features, *args, **kwargs):
        # TODO: assert to make sure only left padding is used?
        # pad audio on the left
        audio_len = torch.LongTensor([f["audio"].shape[-1] for f in features])
        max_audio_len = audio_len.max().item()
        audio = torch.concat(
            [
                torch.nn.functional.pad(
                    f["audio"], (max_audio_len - f["audio"].shape[-1], 0)
                )
                for f in features
            ],
            dim=0,
        )

        return {"audio": audio, "audio_len": audio_len}
