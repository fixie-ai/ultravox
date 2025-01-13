import dataclasses
from typing import List, Optional, Set

import datasets
import simple_parsing
import yaml

from ultravox.tools.ds_tool import caching
from ultravox.tools.ds_tool import ds_commons
from ultravox.tools.ds_tool import tts

tts_client: caching.CachingTtsWrapper


@dataclasses.dataclass
class TtsTask(ds_commons.DSToolTask):
    # Jinja template for the text that needs to be converted to audio
    template: str = simple_parsing.field(alias="-T")
    implementation: str = simple_parsing.field(default="azure", alias="-i")
    json_mode: bool = simple_parsing.field(default=False, alias="-j")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")

    def __post_init__(self):
        # The TTS client is separate from the task to avoid pickling issues when multiprocessing.
        if self.audio_column_name is None:
            self.audio_column_name = f"{self.column_name}_audio"
        global tts_client
        tts_client = caching.CachingTtsWrapper(
            tts.create_client(self.implementation, self.sample_rate),
            implementation=self.implementation,
        )

        if self.template.startswith("@"):
            with open(self.template[1:], "r") as template_file:
                self.template = template_file.read()

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: List[str],
    ) -> datasets.Dataset:
        print(f'TTS mapping "{self.template}" to "{self.audio_column_name}"...')
        ds_split = ds_split.map(
            self._map_sample,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"exclude_fields": exclude_fields},
        )
        column_type = datasets.Audio(sampling_rate=self.sample_rate)
        if self.json_mode and isinstance(
            ds_split.features[self.audio_column_name], datasets.Sequence
        ):
            column_type = datasets.Sequence(column_type)
        return ds_split.cast_column(self.audio_column_name, column_type)

    def _map_sample(self, sample, exclude_fields: Set[str]):
        # using a Jinja template for some added flexibility, template can include variables and functions
        # e.g., {{ text }} or {{ text_proc.format_asr_text(text) }}
        text_or_texts = ds_commons.apply_jinja_template(
            self.template, sample, exclude_fields
        )

        if self.json_mode:
            text_or_texts = yaml.safe_load(text_or_texts)
            assert isinstance(text_or_texts, list)
            assert all(isinstance(turn, str) for turn in text_or_texts)

        sample[self.audio_column_name] = tts_client.tts(text_or_texts, self.voice)
        return sample
