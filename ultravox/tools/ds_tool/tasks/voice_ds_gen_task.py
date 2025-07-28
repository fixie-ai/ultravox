"""
This script is used to generate a set of unique voices from an audio dataset.

Example usage:

1. To generate the voice datasets run:
    export HF_TOKEN=$HF_WRITE_TOKEN
    just ds_tool voice_ds_gen -d mozilla-foundation/common_voice_17_0 -S en -s train -n 900000 --chunk_split_threshold 900001 -u fixie-ai/cv-en-voices --private --shuffle -ods 20000 -l 8 -pk {{client_id}} -T "\"{{sentence}}\""
    just ds_tool voice_ds_gen -d mozilla-foundation/common_voice_17_0 -S zh-CN -s train -n 900000 --chunk_split_threshold 900001 -u fixie-ai/cv-zh-voices --private --shuffle -ods 20000 -l 8 -pk {{client_id}} -T "\"{{sentence}}\""
    just ds_tool voice_ds_gen -d fixie-ai/wenetspeech -s train -S L_fixed -n 300000 --chunk_split_threshold 3000001 -u fixie-ai/wenetspeech-voices --private --shuffle -ods 10000 -l 5 -pk \""{{ segment_id.split('_')[:2] | join('_') }}"\" -T "{{text}}"
    just ds_tool voice_ds_gen -d speechcolab/gigaspeech -S l -s train -u fixie-ai/gs-voices --private --shuffle -ods 16000 -l 8 -pk {{original_full_path}} -T "\"{{text_proc.format_asr_text(text)}}\"" --chunk_split_threshold 10000000
    just ds_tool voice_ds_gen -d fixie-ai/ultravox-calls-deepgram-v2 -S en-prekrisp-high-confidence -s train -u fixie-ai/ultravox-calls-deepgram-v2-voices --private --shuffle -ods 10000 -l 6 -m 15 -T {{transcription}} -pk {{message_id}}

2. To combine all the created datasets:
    just python -m ultravox.tools.ds_tool.tasks.voice_ds_gen_task -d fixie-ai/wenetspeech-voices fixie-ai/cv-en-voices fixie-ai/cv-fr-voices fixie-ai/gs-voices -o fixie-ai/combined-voices
"""

import dataclasses

import datasets
import simple_parsing

from ultravox.tools.ds_tool import ds_commons


@dataclasses.dataclass
class VoiceDSGenerationTask(ds_commons.DSToolTask):
    """
    This task is used to generate voice datasets from text datasets.
    The text dataset should contain a column with the text that might be used as the transcript.
    """

    # The column name containing the text to convert to audio
    text_template: str = simple_parsing.field(alias="-T")
    # The field name to use as unique identifier for the voice
    unique_field: str | None = simple_parsing.field(alias="-pk")
    # The column name to use as audio
    audio_column_name: str = simple_parsing.field(default="audio", alias="-a")
    # Minimum audio length in seconds
    min_audio_length: int = simple_parsing.field(default=6, alias="-l")
    # Maximum audio length in seconds
    max_audio_length: int = simple_parsing.field(default=10, alias="-m")
    # Minimum text length in characters
    min_text_length: int = simple_parsing.field(default=30, alias="-tl")
    # Output dataset size
    dataset_size: int = simple_parsing.field(default=1000, alias="-ods")

    def __post_init__(self):
        self.seen_voices = set()

    def _audio_length_filter(self, sample):
        audio = sample["audio"]
        return (
            self.min_audio_length
            <= audio["array"].shape[-1] / audio["sampling_rate"]
            <= self.max_audio_length
        )

    def _get_unique_indices(self, ds_split):
        indices = []
        for i, sample in enumerate(ds_split):
            if sample["almost_unique_id"] not in self.seen_voices:
                self.seen_voices.add(sample["almost_unique_id"])
                indices.append(i)
                if len(indices) >= self.dataset_size:
                    break
        return indices

    def _text_length_filter(self, sample):
        return len(sample["clean_transcript"]) > self.min_text_length

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: set[str],
    ) -> datasets.Dataset:
        ds_split = ds_split.filter(self._audio_length_filter, num_proc=num_proc)
        ds_split = ds_split.map(
            ds_commons.add_column_with_template,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
            fn_kwargs={
                "column_name": "clean_transcript",
                "template": self.text_template,
                "exclude_fields": exclude_fields,
            },
        )
        ds_split = ds_split.filter(self._text_length_filter, num_proc=num_proc)
        ds_split = ds_split.map(
            ds_commons.add_column_with_template,
            num_proc=num_proc,
            fn_kwargs={
                "column_name": "almost_unique_id",
                "template": self.unique_field,
                "exclude_fields": exclude_fields,
            },
        )
        indices = self._get_unique_indices(ds_split)
        ds_split = ds_split.select(indices)
        return ds_split


@dataclasses.dataclass
class CombineVoiceDSJob:
    output_ds_name: str = simple_parsing.field(alias="-o")
    ds_names: list[str] = simple_parsing.list_field(alias="-d")
    # The sample rate to resample the audio to.
    target_sample_rate: int = simple_parsing.field(default=24000, alias="-r")
    max_audio_length: int = simple_parsing.field(default=10)
    min_audio_length: int = simple_parsing.field(default=4)
    private: bool = simple_parsing.field(default=True, alias="-p")


def combine_voice_datasets(args: CombineVoiceDSJob):
    def filter_audio_length(x):
        len_in_seconds = x["audio"]["array"].shape[-1] / x["audio"]["sampling_rate"]
        return args.min_audio_length < len_in_seconds < args.max_audio_length

    ds_list = [datasets.load_dataset(ds_name)["train"] for ds_name in args.ds_names]

    # filter out samples that are too short or too long
    ds_list = [ds.filter(filter_audio_length, num_proc=32) for ds in ds_list]
    # restrict to only 3 columns: audio, clean_transcript, almost_unique_id
    columns_to_keep = ["audio", "clean_transcript", "almost_unique_id"]
    ds_list = [ds.select_columns(columns_to_keep) for ds in ds_list]
    # resample audio to 24kHz
    ds_list = [
        ds.cast_column("audio", datasets.Audio(sampling_rate=args.target_sample_rate))
        for ds in ds_list
    ]

    # concatenate all datasets
    all_ds = datasets.concatenate_datasets(ds_list)

    # split into train, test, validation, and train_small
    all_ds = all_ds.train_test_split(test_size=1024, shuffle=True)
    all_ds["train"], all_ds["validation"] = (
        all_ds["train"].train_test_split(test_size=1024, shuffle=True).values()
    )
    all_ds["train"], all_ds["train_small"] = (
        all_ds["train"].train_test_split(test_size=1024, shuffle=True).values()
    )

    all_ds.push_to_hub(args.output_ds_name, private=args.private)


if __name__ == "__main__":
    combine_voice_datasets(simple_parsing.parse(CombineVoiceDSJob))
