import dataclasses
import os
from typing import Any, Dict

import datasets
import datasets.distributed
import deepgram

from ultravox.tools.ds_tool import audio_utils
from ultravox.tools.ds_tool import ds_commons


@dataclasses.dataclass
class AsrTask(ds_commons.DSToolTask):
    audio_column: str = "audio"
    text_column: str = "transcript"
    asr_model: str = "nova-2"
    api_key: str = os.environ.get("DEEPGRAM_API_KEY", "")
    language: str | None = None

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is not set")

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: set[str],
    ) -> datasets.Dataset:
        asr_client = deepgram.DeepgramClient(api_key=self.api_key)

        ds_split = ds_split.map(
            self.asr,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"client": asr_client},
        )
        return ds_split

    def asr(self, sample: Dict[str, Any], client: deepgram.DeepgramClient):
        audio = sample[self.audio_column]

        buffer = audio_utils.numpy_audio_to_wav(audio["array"], audio["sampling_rate"])

        payload: deepgram.FileSource = {"buffer": buffer}
        options = deepgram.PrerecordedOptions(
            model=self.asr_model,
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=False,
            language=self.language,
        )

        response = client.listen.rest.v("1").transcribe_file(payload, options)

        sample[self.text_column] = response.to_json(indent=4)
        return sample
