import dataclasses
import os
from concurrent import futures
from typing import Dict, Optional, Union

import datasets
import simple_parsing
import openai


from ultravox.tools import tts


CHAT_CLIENT = openai.Client()
TTS_CLIENT = tts.AzureTts()

DEFAULT_TEXTGEN_TEMPLATE = """Passage: {passage}

Question: {question}

Answer: {answer}

Provide a short explanation to the question given the passage that entails the answer."""


@dataclasses.dataclass(frozen=True)
class TtsTask:
    column_name: str = simple_parsing.field(default="question", alias="-c")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")


@dataclasses.dataclass(frozen=True)
class TextGenerationTask:
    language_model: str = simple_parsing.field(default="gpt-4o", alias="-m")
    new_column_name: str = simple_parsing.field(default="explanation", alias="-c")
    template: str = DEFAULT_TEXTGEN_TEMPLATE


# This script is used to generate audio samples from text using a TTS model.
# Ex: just ds_tool -T tts -d google/boolq -u fixie-ai/boolq-audio -c question -a audio
# Ex: just ds_tool -T textgen -d fixie-ai/boolq-audio -u fixie-ai/boolq-audio -c explanation
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

    token: Optional[str] = simple_parsing.field(default=None, alias="-t")

    task: Union[TtsTask, TextGenerationTask] = simple_parsing.subgroups(
        {"tts": TtsTask, "textgen": TextGenerationTask},
        default_factory=TtsTask,
        alias="-T",
    )


def _tts_sample(sample, col_name: str, audio_col_name: str):
    text = sample[col_name]
    text = text["text"] if isinstance(text, dict) else text
    sample[audio_col_name] = TTS_CLIENT.tts(text)
    return sample


def _tts_split(
    ds_split: datasets.Dataset, task: TtsTask, num_proc: Optional[int] = None
):
    col_name = task.column_name
    audio_col_name = task.audio_column_name or f"{col_name}_audio"
    print(f'TTS mapping "{col_name}" to "{audio_col_name}"...')

    return ds_split.map(
        _tts_sample,
        num_proc=num_proc,
        fn_kwargs={
            "col_name": col_name,
            "audio_col_name": audio_col_name,
            "voice": task.voice,
            "sample_rate": task.sample_rate,
        },
    ).cast_column(audio_col_name, datasets.Audio(sampling_rate=task.sample_rate))


def _text_gen_sample(sample, template: str, col_name: str, language_model: str):
    input_text = template.format(**sample)
    response = CHAT_CLIENT.chat.completions.create(
        model=language_model,
        messages=[{"role": "user", "content": input_text}],
        max_tokens=128,
        temperature=0,
    )
    sample[col_name] = response.choices[0].message.content
    return sample


def _text_gen_split(
    ds_split: datasets.Dataset, task: TextGenerationTask, num_proc: Optional[int] = None
):
    col_name = task.new_column_name
    template = task.template
    print(f'Text gen for column: "{col_name}" with template:\n{template}')
    return ds_split.map(
        _text_gen_sample,
        num_proc=num_proc,
        fn_kwargs={
            "language_model": task.language_model,
            "col_name": col_name,
            "template": template,
        },
    )


def main(args: DatasetToolArgs):
    ds_name = args.dataset_name
    print(f'Loading dataset "{ds_name}" for task "{args.task}"...')
    data_dict = datasets.load_dataset(
        ds_name, args.dataset_subset, split=args.dataset_split
    )

    if args.dataset_split:
        data_dict = {args.dataset_split: data_dict}

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
    hub_args = {
        "config_name": args.dataset_subset,
        "token": token,
        "revision": args.upload_branch,
    }
    if args.num_shards:
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
    print(simple_parsing.parse(DatasetToolArgs))
