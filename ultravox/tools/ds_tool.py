import dataclasses
import os
from typing import Any, Dict, Optional, Union, List

import jmespath
import datasets
import openai
import simple_parsing

from ultravox.tools import tts

chat_client = openai.Client()
tts_client = tts.AzureTts()

DEFAULT_TEXTGEN_TEMPLATE = """Passage: {passage}

Question: {question}

Answer: {answer}

Provide a short explanation to the question given the passage that provides a rationale for the answer."""


@dataclasses.dataclass
class TtsTask:
    # Column name containing the text to convert to audio. It can be a JMESPath expression.
    column_name: str = simple_parsing.field(default="question", alias="-c")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")

    def __post_init__(self):
        if self.audio_column_name is None:
            self.audio_column_name = f"{self.column_name}_audio"

    def map_split(self, ds_split: datasets.Dataset, num_proc: int) -> datasets.Dataset:
        print(f'TTS mapping "{self.column_name}" to "{self.audio_column_name}"...')
        return ds_split.map(self._map_sample, num_proc=num_proc).cast_column(
            self.audio_column_name, datasets.Audio(sampling_rate=self.sample_rate)
        )

    def _map_sample(self, sample):
        text = jmespath.search(sample, self.column_name)
        text = text["text"] if isinstance(text, dict) else text
        sample[self.audio_column_name] = tts_client.tts(text)
        return sample


@dataclasses.dataclass
class TextGenerationTask:
    new_column_name: str = simple_parsing.field(default="explanation", alias="-c")
    template: str = simple_parsing.field(default=DEFAULT_TEXTGEN_TEMPLATE, alias="-T")
    # Interpret the template as a JMESPath expression.
    use_jmespath: bool = simple_parsing.field(default=False, alias="-j")

    language_model: str = simple_parsing.field(default="gpt-4o", alias="-m")
    max_tokens: int = 128
    temperature: float = 0

    def __post_init__(self):
        if self.template.startswith("@"):
            with open(self.template[1:], "r") as template_file:
                self.template = template_file.read()

    def map_split(self, ds_split: datasets.Dataset, num_proc: int) -> datasets.Dataset:
        print(f'Generating "{self.new_column_name}" with template:\n{self.template}')
        return ds_split.map(self._map_sample, num_proc=num_proc)

    @staticmethod
    def get_messages_ending_with_user(turn_contents: List[str]):
        roles = ["user", "assistant"]
        if len(turn_contents) % 2 == 0:
            roles = roles[::-1]
        return [
            {"role": roles[i % 2], "content": content}
            for i, content in enumerate(turn_contents)
        ]

    def _map_sample(self, sample):
        if self.use_jmespath:
            turns = jmespath.search(sample, self.template)
        else:
            turns = [self.template.format(**sample)]

        response = chat_client.chat.completions.create(
            model=self.language_model,
            messages=self.get_messages_ending_with_user(turns),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        sample[self.new_column_name] = response.choices[0].message.content
        return sample


# This script is used to either generate audio samples from text using a TTS model, or to generate text samples using a text generation model.
# Ex: just ds_tool tts -d google/boolq -u fixie-ai/boolq-audio -c question -a audio --token $HF_WRITE_TOKEN
# Ex: just ds_tool textgen -d fixie-ai/boolq-audio -u fixie-ai/boolq-audio -c explanation
# Ex: just ds_tool textgen -d ylacombe/expresso -u fixie-ai/expresso -c continuation -T @expresso_template.txt
# Ex: just ds_tool textgen -d allenai/soda --shuffle True --split train -n 10000 -u fixie-ai/soda -c alt_last_turn -j -m llama3-8b -T "dialogue[:-1]"
# Ex: just ds_tool textgen -d allenai/soda --shuffle True --split validation -n 1000 -u fixie-ai/soda -c alt_last_turn -j -m llama3-8b -T "dialogue[:-1]"
# Ex: just ds_tool textgen -d allenai/soda --shuffle True --split test -n 1000 -u fixie-ai/soda -c alt_last_turn -j -m llama3-8b -T "dialogue[:-1]"
# Ex: just ds_tool textgen -d fixie-ai/soda -u fixie-ai/soda -c "dialogue[-2]" -a audio_one_but_last
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
    upload_branch: Optional[str] = simple_parsing.field(default="main", alias="-b")
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
            data_dict[args.dataset_split].push_to_hub(args.upload_name, **hub_args)
        else:
            data_dict.push_to_hub(args.upload_name, **hub_args)
    except Exception as e:
        print(f"Failed to push to hub: {e}")

        # If the push fails or upload_name is not specified, save the data locally.
        for split in data_dict.keys():
            output_name = f"{split}-00000-of-00001.parquet"
            data_dict[split].to_parquet(output_name)
            print(f"Saved to {output_name}")


if __name__ == "__main__":
    main(simple_parsing.parse(DatasetToolArgs))
