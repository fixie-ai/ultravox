import dataclasses
import os
from typing import Any, Dict, Optional, Union

import datasets
import openai
import simple_parsing

from ultravox.tools import tts

chat_client = openai.Client()
tts_client = tts.AzureTts()

DEFAULT_TEXTGEN_TEMPLATE = """Passage: {passage}

Question: {question}

Answer: {answer}

Provide a short explanation to the question given the passage that entails the answer."""


@dataclasses.dataclass
class TtsTask:
    column_name: str = simple_parsing.field(default="question", alias="-c")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")

    def __post_init__(self):
        if self.audio_column_name is None:
            self.audio_column_name = f"{self.column_name}_audio"


@dataclasses.dataclass
class TextGenerationTask:
    new_column_name: str = simple_parsing.field(default="explanation", alias="-c")
    template: str = simple_parsing.field(default=DEFAULT_TEXTGEN_TEMPLATE, alias="-T")

    language_model: str = simple_parsing.field(default="gpt-4o", alias="-m")
    max_tokens: int = 128
    temperature: float = 0


# This script is used to either generate audio samples from text using a TTS model, or to generate text samples using a text generation model.
# Ex: just ds_tool -t tts -d google/boolq -u fixie-ai/boolq-audio -c question -a audio --token $HF_WRITE_TOKEN
# Ex: just ds_tool -t textgen -d fixie-ai/boolq-audio -u fixie-ai/boolq-audio -c explanation
# Ex: just ds_tool -t textgen -d ylacombe/expresso -u fixie-ai/expresso -c continuation \
#         -T "\"Continue the following sentence in a way that reflects a ‘{style}’ tone in a coherent style:\n{text}\""
@dataclasses.dataclass
class DatasetToolArgs:
    dataset_name: str = simple_parsing.field(alias="-d")
    dataset_subset: str = simple_parsing.field(default="default", alias="-S")
    dataset_split: Optional[str] = simple_parsing.field(default=None, alias="-s")

    num_samples: Optional[int] = simple_parsing.field(default=None, alias="-n")
    num_workers: int = simple_parsing.field(default=16, alias="-w")

    upload_name: Optional[str] = simple_parsing.field(default=None, alias="-u")
    upload_branch: Optional[str] = simple_parsing.field(default="main", alias="-b")
    num_shards: Optional[int] = simple_parsing.field(default=None, alias="-N")

    token: Optional[str] = None

    task: Union[TtsTask, TextGenerationTask] = simple_parsing.subgroups(
        {"tts": TtsTask, "textgen": TextGenerationTask},  # type: ignore
        default_factory=TtsTask,
        alias="-t",
    )


def _tts_split(ds_split: datasets.Dataset, task: TtsTask, num_proc: int):
    def _tts_sample(sample):
        text = sample[task.column_name]
        text = text["text"] if isinstance(text, dict) else text
        sample[task.audio_column_name] = tts_client.tts(text)
        return sample

    print(f'TTS mapping "{task.column_name}" to "{task.audio_column_name}"...')

    return ds_split.map(_tts_sample, num_proc=num_proc).cast_column(
        task.audio_column_name, datasets.Audio(sampling_rate=task.sample_rate)
    )


def _text_gen_split(
    ds_split: datasets.Dataset, task: TextGenerationTask, num_proc: int
):
    def _text_gen_sample(sample):
        input_text = task.template.format(**sample)
        response = chat_client.chat.completions.create(
            model=task.language_model,
            messages=[{"role": "user", "content": input_text}],
            max_tokens=task.max_tokens,
            temperature=task.temperature,
        )
        sample[task.new_column_name] = response.choices[0].message.content
        return sample

    print(
        f'Text gen for column: "{task.new_column_name}" with template:\n{task.template}'
    )
    return ds_split.map(_text_gen_sample, num_proc=num_proc)


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
        if args.num_samples:
            ds_split = ds_split.select(range(args.num_samples))

        if isinstance(args.task, TtsTask):
            data_dict[split] = _tts_split(
                ds_split, args.task, num_proc=args.num_workers
            )

        elif isinstance(args.task, TextGenerationTask):
            data_dict[split] = _text_gen_split(
                ds_split, args.task, num_proc=args.num_workers
            )

    token = args.token or os.environ.get("HF_TOKEN")
    hub_args: Dict[str, Any] = {
        "config_name": args.dataset_subset,
        "token": token,
        "revision": args.upload_branch,
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
