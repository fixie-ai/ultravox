import dataclasses
import json
import os
from typing import Any, Dict, Optional, Union

import datasets
import jinja2
import openai
import simple_parsing

from ultravox.tools import tts
from ultravox.tools import wrappers

tts_client: wrappers.CachingTtsWrapper
chat_client: wrappers.CachingChatWrapper


@dataclasses.dataclass
class TtsTask:
    implementation: str = simple_parsing.field(default="azure", alias="-i")
    # Column name containing the text to convert to audio. It can be a Jinja variable expression.
    column_name: str = simple_parsing.field(default="question", alias="-c")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")

    def __post_init__(self):
        # The TTS client is separate from the task to avoid pickling issues when multiprocessing.
        global tts_client
        if self.audio_column_name is None:
            self.audio_column_name = f"{self.column_name}_audio"
        tts_client = wrappers.CachingTtsWrapper(
            tts.create_client(self.implementation, self.sample_rate),
            provider=self.implementation,
        )

    def map_split(self, ds_split: datasets.Dataset, num_proc: int) -> datasets.Dataset:
        print(f'TTS mapping "{self.column_name}" to "{self.audio_column_name}"...')
        return ds_split.map(self._map_sample, num_proc=num_proc).cast_column(
            self.audio_column_name, datasets.Audio(sampling_rate=self.sample_rate)
        )

    def _map_sample(self, sample):
        # using a Jinja template for some added flexibility
        # The {{ var }} syntax is how Jinja denotes variables
        text = jinja2.Template("{{" + self.column_name + "}}").render(**sample)
        text = text["text"] if isinstance(text, dict) else text
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

    def __post_init__(self):
        # The OAI client is separate from the task to avoid pickling issues when multiprocessing.
        global chat_client
        # Caching the client to avoid repeated calls to the API if the tool fails.
        chat_client = wrappers.CachingChatWrapper(
            openai.Client(base_url=self.base_url, api_key=self.api_key),
            base_url=self.base_url,
        )
        if self.template.startswith("@"):
            with open(self.template[1:], "r") as template_file:
                self.template = template_file.read()

    def map_split(self, ds_split: datasets.Dataset, num_proc: int) -> datasets.Dataset:
        print(f'Generating "{self.new_column_name}" with template:\n{self.template}')
        return ds_split.map(self._map_sample, num_proc=num_proc)

    def _map_sample(self, sample):
        rendered = jinja2.Template(self.template).render(**sample, json_dump=json.dumps)

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
@dataclasses.dataclass
class DatasetToolArgs:
    dataset_name: str = simple_parsing.field(alias="-d")
    dataset_subset: Optional[str] = simple_parsing.field(default=None, alias="-S")
    dataset_split: Optional[str] = simple_parsing.field(default=None, alias="-s")

    shuffle: bool = simple_parsing.field(default=False)
    shuffle_seed: int = simple_parsing.field(default=42)
    num_samples: Optional[int] = simple_parsing.field(default=None, alias="-n")
    num_workers: int = simple_parsing.field(default=16, alias="-w")

    upload_name: Optional[str] = simple_parsing.field(default=None, alias="-u")
    upload_branch: Optional[str] = simple_parsing.field(default="main", alias="-B")
    num_shards: Optional[int] = simple_parsing.field(default=None, alias="-N")
    private: bool = simple_parsing.field(default=False)

    token: Optional[str] = None

    task: Union[TtsTask, TextGenerationTask] = simple_parsing.subgroups(
        {"tts": TtsTask, "textgen": TextGenerationTask},  # type: ignore
        default_factory=TtsTask,
        positional=True,
    )


def main(args: DatasetToolArgs):
    ds_name = args.dataset_name
    print(f'Loading dataset "{ds_name}" for task {args.task}')
    data_dict: datasets.DatasetDict = datasets.load_dataset(
        ds_name, args.dataset_subset, split=args.dataset_split
    )
    if args.dataset_split:
        data_dict = datasets.DatasetDict(**{args.dataset_split: data_dict})

    for split, ds_split in data_dict.items():
        print(f'Processing split "{split}"...')
        if args.shuffle:
            ds_split = ds_split.shuffle(seed=args.shuffle_seed)
        if args.num_samples:
            ds_split = ds_split.select(range(args.num_samples))
        data_dict[split] = args.task.map_split(ds_split, args.num_workers)

    token = args.token or os.environ.get("HF_TOKEN")
    hub_args: Dict[str, Any] = {
        "config_name": args.dataset_subset or "default",
        "token": token,
        "revision": args.upload_branch,
        "private": args.private,
    }
    if args.num_shards is not None:
        hub_args["num_shards"] = {split: args.num_shards for split in data_dict.keys()}

    try:
        if args.dataset_split:
            data_dict[args.dataset_split].push_to_hub(
                args.upload_name, split=args.dataset_split, **hub_args
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
