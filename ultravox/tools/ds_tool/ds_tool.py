import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

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
        print(f'Generating "{self.new_column_name}" with template:\n{self.template}')
        return ds_split.map(
            lambda sample: self._map_sample(sample, set(exclude_fields)),
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
        )

    def _map_sample(self, sample, exclude_fields):
        # using a Jinja template for some added flexibility, template can include variables and functions
        # e.g., {{ text }} or {{ text_proc.format_asr_text(text) }}
        try:
            # We need to filter out the audio before the sample is passed into the jinja template
            # or it will get loaded into memory and spike usage.
            filtered_sample = {
                k: sample[k] for k in sample.keys() if k not in exclude_fields
            }

            rendered = jinja2.Template(
                self.template, undefined=jinja2.StrictUndefined
            ).render(**filtered_sample, json_dump=json.dumps, text_proc=text_proc)
        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"template: {self.template}")
            print(f"sample keys: {list(filtered_sample.keys())}")
            raise ValueError(
                f"Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e

        if self.json_mode:
            turns = yaml.safe_load(rendered)
            assert isinstance(turns, list)
            assert all(isinstance(turn, dict) for turn in turns)
            assert len(turns) > 0
            assert turns[-1].get("role", None) == "user"
        else:
            turns = [{"role": "user", "content": rendered}]


        max_retries = 3  # Number of retry attempts
        retry_delay = 2  # Delay in seconds between retries

        for attempt in range(max_retries):
            try:
                start_time = time.time()  # Record start time
                sample[self.new_column_name] = chat_client.chat_completion(
                    model=self.language_model,
                    messages=turns,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                break
            except Exception as e:
                duration = time.time() - start_time 
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed after {duration:.2f} seconds: {e}.")
                    print(f"Message: {turns}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Attempt {attempt + 1} failed after {duration:.2f} seconds: {e}.")
                    print(f"Message: {turns}")
                    print("No more retries left.")
                    raise

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
    dataset_subset: Optional[str] = simple_parsing.field(default=None, alias="-S")
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

    for split, ds_split in data_dict.items():
        print(
            f"Processing dataset: {ds_name}, subset {args.dataset_subset}, split {args.dataset_split}, containing {len(ds_split)} samples"
        )
        if args.shuffle:
            ds_split = ds_split.shuffle(seed=args.shuffle_seed)
        if args.num_samples:
            ds_split = ds_split.select(range(args.num_samples))
        data_dict[split] = args.task.map_split(
            ds_split, args.num_workers, args.writer_batch_size, args.exclude_fields
        )

    hub_args: Dict[str, Any] = {
        "config_name": args.upload_subset or "default",
        "token": args.token or os.environ.get("HF_TOKEN"),
        "revision": args.upload_branch,
        "private": args.private,
    }

    if args.num_shards is not None:
        hub_args["num_shards"] = args.num_shards

    try:
        if args.dataset_split:
            data_dict[args.dataset_split].push_to_hub(
                args.upload_name, split=args.upload_split, **hub_args
            )
        else:
            data_dict.push_to_hub(args.upload_name, **hub_args)
    except Exception as e:
        print(f"Failed to push to hub: {e}")

        # If the push fails or upload_name is not specified, save the data locally.
        for split in data_dict.keys():
            output_name = f"{split}-00000-of-00001.parquet"
            data_dict[split].to_parquet(output_name)
            print(f"Saved to {output_name}")
            print(f"Sample {0} of {split}: {data_dict[split][0]}")


if __name__ == "__main__":
    main(simple_parsing.parse(DatasetToolArgs))
