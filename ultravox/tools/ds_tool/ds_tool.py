import dataclasses
import json
import os
from typing import Any, Dict, Optional, Union, List

import datasets
import jinja2
import openai
import simple_parsing

from ultravox.data import text_proc
from ultravox.tools.ds_tool import caching
from ultravox.tools.ds_tool import tts

from ultravox.data.text_proc import format_asr_text

tts_client: caching.CachingTtsWrapper
chat_client: caching.CachingChatWrapper


@dataclasses.dataclass
class TtsTask:
    implementation: str = simple_parsing.field(default="azure", alias="-i")
    # Column name containing the text to convert to audio. It can be a Jinja variable expression.
    column_name: str = simple_parsing.field(default="question", alias="-c")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")
    write_batch_size: int = 1000
    format_fields: List[str] = simple_parsing.field(default_factory=list)

    def __post_init__(self):
        # The TTS client is separate from the task to avoid pickling issues when multiprocessing.
        global tts_client
        if self.audio_column_name is None:
            self.audio_column_name = f"{self.column_name}_audio"
        tts_client = caching.CachingTtsWrapper(
            tts.create_client(self.implementation, self.sample_rate),
            provider=self.implementation,
        )

    def map_split(
        self, ds_split: datasets.Dataset, num_proc: int, writer_batch_size: int
    ) -> datasets.Dataset:
        print(f'TTS mapping "{self.column_name}" to "{self.audio_column_name}"...')
        return ds_split.map(
            self._map_sample, num_proc=num_proc, writer_batch_size=writer_batch_size
        ).cast_column(
            self.audio_column_name, datasets.Audio(sampling_rate=self.sample_rate)
        )

    def _map_sample(self, sample):
        for field in self.format_fields:
            sample[field] = format_asr_text(sample[field])
        # using a Jinja template for some added flexibility
        # The {{ var }} syntax is how Jinja denotes variables
        try:
            text = jinja2.Template(
                "{{" + self.column_name + "}}", undefined=jinja2.StrictUndefined
            ).render(**sample)
        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"column_name: {self.column_name}")
            print(f"sample keys: {list(sample.keys())}")
            raise ValueError(
                f"Template rendering failed. Make sure column_name exists in the sample."
            ) from e

        sample[self.audio_column_name] = tts_client.tts(text, self.voice)
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
    write_batch_size: int = 1000
    format_fields: List[str] = simple_parsing.field(default_factory=list)

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
        self, ds_split: datasets.Dataset, num_proc: int, writer_batch_size: int
    ) -> datasets.Dataset:
        print(f'Generating "{self.new_column_name}" with template:\n{self.template}')
        return ds_split.map(
            self._map_sample, num_proc=num_proc, writer_batch_size=writer_batch_size
        )

    def _map_sample(self, sample):
        # using a Jinja template for some added flexibility, template can include variables and functions
        # e.g., {{ text }} or {{ text_proc.format_asr_text(text) }}
        try:
            rendered = jinja2.Template(
                self.template, undefined=jinja2.StrictUndefined
            ).render(**sample, json_dump=json.dumps, text_proc=text_proc)
        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"template: {self.template}")
            print(f"sample keys: {list(sample.keys())}")
            raise ValueError(
                f"Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e

        if self.json_mode:
            turns = json.loads(rendered)
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
#   just ds_tool tts -d google/boolq -u fixie-ai/boolq-audio -c question -a audio --token $HF_WRITE_TOKEN
#   just ds_tool textgen -d fixie-ai/boolq-audio -u fixie-ai/bar -c explanation -b https://api.fireworks.ai/inference/v1 -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct
#   just ds_tool textgen -d ylacombe/expresso -u fixie-ai/expresso -c continuation -T @expresso_template.txt
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
    ds: datasets.DatasetDict = datasets.load_dataset(
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
            ds_split, args.num_workers, args.writer_batch_size
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
             print(f"Sample {0} of {args.upload_subset}: {ds[0]}")


if __name__ == "__main__":
    main(simple_parsing.parse(DatasetToolArgs))
