import dataclasses
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import jinja2
import openai
import simple_parsing
import yaml

from ultravox.data import text_proc
from ultravox.tools.ds_tool import caching
from ultravox.tools.ds_tool import tts

tts_client: caching.CachingTtsWrapper
chat_client: caching.CachingChatWrapper


@dataclasses.dataclass
class TtsTask:
    # Jinja template for the text that needs to be converted to audio
    template: str = simple_parsing.field(alias="-T")
    implementation: str = simple_parsing.field(default="azure", alias="-i")
    json_mode: bool = simple_parsing.field(default=False, alias="-j")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")

    def __post_init__(self):
        # The TTS client is separate from the task to avoid pickling issues when multiprocessing.
        global tts_client
        if self.audio_column_name is None:
            self.audio_column_name = f"{self.column_name}_audio"
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
            self._map_sample, num_proc=num_proc, writer_batch_size=writer_batch_size
        )
        column_type = datasets.Audio(sampling_rate=self.sample_rate)
        if self.json_mode and isinstance(
            ds_split.features[self.audio_column_name], datasets.Sequence
        ):
            column_type = datasets.Sequence(column_type)
        return ds_split.cast_column(self.audio_column_name, column_type)

    def _map_sample(self, sample):
        # using a Jinja template for some added flexibility, template can include variables and functions
        # e.g., {{ text }} or {{ text_proc.format_asr_text(text) }}
        try:
            text_or_texts = jinja2.Template(
                self.template, undefined=jinja2.StrictUndefined
            ).render(**sample, json_dump=json.dumps, text_proc=text_proc)
        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"template: {self.template}")
            print(f"sample keys: {list(sample.keys())}")
            raise ValueError(
                f"Template rendering failed. Make sure column_name exists in the sample."
            ) from e

        if self.json_mode:
            text_or_texts = yaml.safe_load(text_or_texts)
            assert isinstance(text_or_texts, list)
            assert all(isinstance(turn, str) for turn in text_or_texts)

        sample[self.audio_column_name] = tts_client.tts(text_or_texts, self.voice)
        return sample


@dataclasses.dataclass
class TextGenerationTask:
    new_column_name: str = simple_parsing.field(alias="-c")
    template: str = simple_parsing.field(alias="-T")
    json_mode: bool = simple_parsing.field(default=False, alias="-j")

    language_model: str = simple_parsing.field(default="gpt-4o", alias="-m")
    base_url: Optional[str] = simple_parsing.field(default=None, alias="-b")
    api_key: Optional[str] = simple_parsing.field(default=None, alias="-k")
    max_tokens: int = 128
    temperature: float = 0

    def __post_init__(self):
        # The OAI client is separate from the task to avoid pickling issues when multiprocessing.
        global chat_client
        # Caching the client to avoid repeated calls to the API if the tool fails.
        chat_client = caching.CachingChatWrapper(
            openai.Client(base_url=self.base_url, api_key=self.api_key),
            unique_id=f"{self.base_url}__{self.language_model}",
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
        # print(f'Generating "{self.new_column_name}" with template:\n{self.template}')
        ds_mapped = ds_split.map(
            lambda sample: self._map_sample(sample, set(exclude_fields)),
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
        )

        # Filter out samples where new_column_name is None
        return ds_mapped.filter(
            lambda sample: sample[self.new_column_name] != None,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
        )

    def _map_sample(self, sample, exclude_fields):
        # using a Jinja template for some added flexibility, template can include variables and functions
        # e.g., {{ text }} or {{ text_proc.format_asr_text(text) }}
        try:
            # Filter out the audio before the sample is passed into the jinja template, or it will get loaded into memory.
            filtered_sample = {
                k: sample[k] for k in sample.keys() if k not in exclude_fields
            }
            rendered = jinja2.Template(
                self.template, undefined=jinja2.StrictUndefined
            ).render(**filtered_sample, json_dump=json.dumps, text_proc=text_proc)
        except Exception as e:
            if isinstance(e, text_proc.GarbageUtteranceError):
                print("Formatted text is empty. Setting output to None.")
                sample[self.new_column_name] = None
                return sample
            elif isinstance(e, text_proc.EmptyTranscriptError):
                print("Empty transcript after processing. Setting output to None.")
                sample[self.new_column_name] = None
                return sample
            elif isinstance(e, jinja2.TemplateError):
                print(f"Error rendering template: {e}")
                print(f"template: {self.template}")
                print(f"sample keys: {list(filtered_sample.keys())}")
                raise ValueError(
                    f"Template rendering failed. Make sure all keys in the template exist in the sample."
                ) from e
            else:
                raise e

        if self.json_mode:
            turns = yaml.safe_load(rendered)
            assert isinstance(turns, list)
            assert all(isinstance(turn, dict) for turn in turns)
            assert len(turns) > 0
            assert turns[-1].get("role", None) == "user"
        else:
            turns = [{"role": "user", "content": rendered}]

        sample[self.new_column_name] = chat_client.chat_completion(
            model=self.language_model,
            messages=turns,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return sample


# This script is used to either generate audio samples from text using a TTS model, or to generate text samples using a text generation model.
#   just ds_tool tts -d google/boolq -u fixie-ai/boolq-audio -T {{question}} -a audio --token $HF_WRITE_TOKEN
#   just ds_tool textgen -d fixie-ai/boolq-audio -u fixie-ai/bar -T {{explanation}} -b https://api.fireworks.ai/inference/v1 -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct
#   just ds_tool textgen -d ylacombe/expresso -u fixie-ai/expresso -T {{continuation}} -T @expresso_template.txt
#   just ds_tool textgen --new_column_name continuation --dataset_name openslr/librispeech_asr --dataset_subset clean --dataset_split train.360 \
#        --shuffle --upload_name fixie-ai/librispeech_asr --private --base_url https://api.fireworks.ai/inference/v1 \
#        --api_key $FIREWORKS_API_KEY --token $HF_TOKEN --language_model accounts/fireworks/models/llama-v3-8b-instruct \
#        --template @ultravox/tools/ds_tool/continuation.jinja --max_tokens 64 --num_workers 30 --writer_batch_size 30
@dataclasses.dataclass
class DatasetToolArgs:
    # HF source dataset parameters
    dataset_name: str = simple_parsing.field(alias="-d")
    dataset_subset: Optional[str] = simple_parsing.field(default="default", alias="-S")
    dataset_split: Optional[str] = simple_parsing.field(default=None, alias="-s")

    # Local processing parameters
    shuffle: bool = simple_parsing.field(default=False)
    shuffle_seed: int = simple_parsing.field(default=42)
    num_samples: Optional[int] = simple_parsing.field(default=None, alias="-n")
    num_workers: int = simple_parsing.field(default=16, alias="-w")
    writer_batch_size: int = simple_parsing.field(default=1000)
    exclude_fields: List[str] = simple_parsing.field(default_factory=lambda: ["audio"])

    # HF destination dataset parameters
    upload_name: Optional[str] = simple_parsing.field(default=None, alias="-u")
    upload_branch: Optional[str] = simple_parsing.field(default="main", alias="-B")
    # eg if the original split="train", but we want to upload it as "validation"
    upload_subset: Optional[str] = simple_parsing.field(default=None)
    upload_split: Optional[str] = simple_parsing.field(default=None)
    num_shards: Optional[int] = simple_parsing.field(default=None, alias="-N")
    private: bool = simple_parsing.field(default=False)
    token: Optional[str] = None

    # Chunk processing parameters
    max_chunk_split: int = simple_parsing.field(default=10)
    chunk_split_threshold: int = simple_parsing.field(default=50000)

    task: Union[TtsTask, TextGenerationTask] = simple_parsing.subgroups(
        {"tts": TtsTask, "textgen": TextGenerationTask},  # type: ignore
        default_factory=TtsTask,
        positional=True,
    )

    def __post_init__(self):
        if not self.upload_subset and self.dataset_subset:
            self.upload_subset = self.dataset_subset
        if self.dataset_split and not self.upload_split:
            self.upload_split = self.dataset_split


class DatasetChunkProcessor:
    args: DatasetToolArgs
    cache_dir: str = ".cache/ds_tool/processed_datasets"
    chunks_not_uploaded: List[Tuple[int, int]] = []
    total_samples_processed: Dict[str, int] = {}

    def __init__(self, args: DatasetToolArgs):
        self.args = args

    def process_and_upload_split(self, split_name: str, ds_split: datasets.Dataset):
        failed_chunk_ranges = self._dynamic_chunk(
            split_name, ds_split, 0, len(ds_split)
        )

        while len(failed_chunk_ranges) > 0:
            new_failed_ranges = []
            for start, end in failed_chunk_ranges:
                print(f"Retrying failed chunk range [{start}, {end})")
                new_failed_ranges.extend(
                    self._dynamic_chunk(split_name, ds_split, start, end)
                )
            failed_chunk_ranges = new_failed_ranges
        print(f"Could not upload chunks: {self.chunks_not_uploaded}")
        print(
            f"Finished processing and uploading all chunks for split {split_name}. Total samples processed: {self.total_samples_processed}"
        )

    def _dynamic_chunk(
        self,
        split_name: str,
        ds_split: datasets.Dataset,
        start_index: int,
        end_index: int,
    ):
        original_chunk_size = end_index - start_index
        if original_chunk_size < self.args.chunk_split_threshold:
            total_chunks = 1
            chunk_size = original_chunk_size
            print(
                f"Chunk range [{start_index}, {end_index}) is too small to split further. Processing and uploading as a single chunk."
            )
        else:
            total_chunks = self.args.max_chunk_split
            chunk_size = math.ceil(original_chunk_size / total_chunks)
        failed_chunk_ranges = []
        print(
            f"Processing and uploading {total_chunks} chunks for range [{start_index}, {end_index}) with chunk size {chunk_size}"
        )
        for i in range(start_index, end_index, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, start_index + original_chunk_size)

            ds_chunk = ds_split.select(range(chunk_start, chunk_end))
            ds_chunk_name = f"chunk-range-{chunk_start:09d}-{chunk_end:09d}.parquet"
            ds_chunk_hub_path = os.path.join(
                str(self.args.upload_subset), split_name, ds_chunk_name
            )
            ds_chunk_cache_path = os.path.join(
                self.cache_dir,
                self.args.dataset_name.replace("/", "__"),
                str(self.args.upload_subset),
                split_name,
                ds_chunk_name,
            )
            try:
                if os.path.exists(ds_chunk_cache_path):
                    print(
                        f"Skipping chunk {ds_chunk_name} as it has already been processed and uploaded."
                    )
                    ds_chunk_processed = datasets.Dataset.from_parquet(
                        ds_chunk_cache_path
                    )
                else:
                    print(f"Processing chunk {ds_chunk_name}")
                    ds_chunk_processed = self._process(ds_chunk)
                    self._upload(ds_chunk_processed, ds_chunk_hub_path, split_name)
                    ds_chunk_processed.to_parquet(ds_chunk_cache_path)
                self.total_samples_processed[split_name] = (
                    self.total_samples_processed.get(split_name, 0)
                    + len(ds_chunk_processed)
                )

            except Exception as e:
                print(f"Failed to upload chunk {ds_chunk_name}: {e}. Retrying later.")
                if total_chunks == 1:
                    print(
                        f"Finished processing and uploading 0/1 chunks for range [{start_index}, {end_index})"
                    )
                    self.chunks_not_uploaded.append((start_index, end_index))
                    return []
                failed_chunk_ranges.append((chunk_start, chunk_end))
        successful_chunks = self.args.max_chunk_split - len(failed_chunk_ranges)
        print(
            f"Finished processing and uploading {successful_chunks}/{self.args.max_chunk_split} chunks for range [{start_index}, {end_index})"
        )
        return failed_chunk_ranges

    def _process(self, ds_chunk: datasets.Dataset) -> datasets.Dataset:
        return self.args.task.map_split(
            ds_chunk,
            self.args.num_workers,
            self.args.writer_batch_size,
            self.args.exclude_fields,
        )

    def _upload(self, ds_chunk_processed: datasets.Dataset, data_dir: str, split_name):
        print(f"Uploading chunk to hub: {data_dir}")
        hub_args: Dict[str, Any] = {
            "config_name": self.args.upload_subset,
            "token": self.args.token or os.environ.get("HF_TOKEN"),
            "private": self.args.private,
            "data_dir": data_dir,
            "num_shards": self.args.num_shards,
            "split": split_name,
        }
        ds_chunk_processed.push_to_hub(self.args.upload_name, **hub_args)


def main(args: DatasetToolArgs):
    ds_name = args.dataset_name
    print(f'Loading dataset "{ds_name}" for task {args.task}')
    data_dict: datasets.DatasetDict = datasets.load_dataset(
        ds_name, args.dataset_subset, split=args.dataset_split
    )

    if isinstance(data_dict, datasets.Dataset):
        data_dict = datasets.DatasetDict({args.upload_split: data_dict})

    if len(data_dict) > 1 and args.upload_split:
        raise ValueError("Cannot upload multiple splits to a single split")

    ds_chunk_proc = DatasetChunkProcessor(args)

    for split_name, ds_split in data_dict.items():
        print(
            f"Processing dataset: {args.dataset_name}, subset {args.dataset_subset}, split {split_name}, containing {len(ds_split)} samples"
        )
        if args.shuffle:
            ds_split = ds_split.shuffle(seed=args.shuffle_seed)
        if args.num_samples:
            ds_split = ds_split.select(range(args.num_samples))

        ds_chunk_proc.process_and_upload_split(split_name, ds_split)


if __name__ == "__main__":
    main(simple_parsing.parse(DatasetToolArgs))
