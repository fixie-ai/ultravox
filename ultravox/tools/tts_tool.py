import dataclasses
import os
from typing import Dict, Optional, Union

import datasets
import simple_parsing

from ultravox.tools import tts


# This script is used to generate audio samples from text using a TTS model.
# Ex: just tts -d google/boolq -c question -a audio -u fixie-ai/boolq-audio
@dataclasses.dataclass
class TtsArgs:
    dataset_name: str = simple_parsing.field(alias="-d")
    dataset_subset: Optional[str] = simple_parsing.field(default=None, alias="-S")
    dataset_split: Optional[str] = simple_parsing.field(default=None, alias="-s")
    column_name: str = simple_parsing.field(default="question", alias="-c")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    num_samples: Optional[int] = simple_parsing.field(default=None, alias="-n")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")
    upload_name: Optional[str] = simple_parsing.field(default=None, alias="-u")
    token: Optional[str] = simple_parsing.field(default=None, alias="-t")


def _tts_split(
    tts_client: tts.AzureTts,
    ds_split: datasets.IterableDataset,
    col_name: str,
    audio_col_name: str,
):
    def get_text(val: Union[str, Dict[str, str]]) -> str:
        return val["text"] if isinstance(val, dict) else val

    def tts_batch(batch):
        batch[audio_col_name] = [
            {"bytes": tts_client.tts(get_text(val))} for val in batch[col_name]
        ]
        return batch

    return ds_split.map(tts_batch, batched=True).cast_column(
        audio_col_name, datasets.Audio(sampling_rate=tts_client._sample_rate)
    )


def main(args: TtsArgs):
    ds_name = args.dataset_name
    col_name = args.column_name
    audio_col_name = args.audio_column_name or f"{col_name}_audio"
    tts_client = tts.AzureTts(voice=args.voice, sample_rate=args.sample_rate)

    print(f'Loading dataset "{ds_name}", mapping "{col_name}" to "{audio_col_name}"...')
    data_dict = datasets.load_dataset(
        ds_name, args.dataset_subset, split=args.dataset_split
    )
    if args.dataset_split:
        data_dict = {args.dataset_split: data_dict}
    for split, ds_split in data_dict.items():
        print(f'Processing split "{split}"...')
        if args.num_samples:
            ds_split = ds_split.select(range(args.num_samples))
        new_split = _tts_split(tts_client, ds_split, col_name, audio_col_name)

        if not args.upload_name:
            output_name = f"{split}-00000-of-00001.parquet"
            new_split.to_parquet(output_name)
        else:
            token = args.token or os.environ.get("HF_TOKEN")
            new_split.push_to_hub(
                args.upload_name,
                config_name=args.dataset_subset,
                split=split,
                token=token,
            )


if __name__ == "__main__":
    main(simple_parsing.parse(TtsArgs))
