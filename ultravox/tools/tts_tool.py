import argparse
import os

import datasets

from ultravox.tools import tts

# This script is used to generate audio samples from text using a TTS model.
# Ex: python tts_tool.py -d google/boolq -c question -a audio -u "fixie-ai/boolq-audio
parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", "-d", type=str, required=True)
parser.add_argument("--dataset-split", "-s", type=str)
parser.add_argument("--column-name", "-c", type=str, default="question")
parser.add_argument("--audio-column-name", "-a", type=str)
parser.add_argument("--num-samples", "-n", type=int)
parser.add_argument("--voice", "-V", type=str)
parser.add_argument("--sample-rate", "-r", type=int, default=16000)
parser.add_argument("--upload-name", "-u", type=str)
parser.add_argument("--token", "-t", type=str)


def _tts_split(
    tts_client: tts.AzureTts,
    ds_split: datasets.IterableDataset,
    col_name: str,
    audio_col_name: str,
):
    def _tts_batch(batch):
        batch[audio_col_name] = [
            {"bytes": tts_client.tts(text)} for text in batch[col_name]
        ]
        return batch

    return ds_split.map(_tts_batch, batched=True).cast_column(
        audio_col_name, datasets.Audio(sampling_rate=tts_client._sample_rate)
    )


def main(args: argparse.Namespace):
    ds_name = args.dataset_name
    col_name = args.column_name
    audio_col_name = args.audio_column_name or f"{col_name}_audio"

    print(f'Loading dataset "{ds_name}", mapping "{col_name}" to "{audio_col_name}"...')
    ds = datasets.load_dataset(ds_name)
    tts_client = tts.AzureTts(voice=args.voice, sample_rate=args.sample_rate)
    for split, ds_split in ds.items():
        if args.dataset_split and split != args.dataset_split:
            continue

        print(f'Processing split "{split}"...')
        if args.num_samples:
            ds_split = ds_split.select(range(args.num_samples))
        new_split = _tts_split(tts_client, ds_split, col_name, audio_col_name)

        if not args.upload_name:
            output_name = f"{split}-00000-of-00001.parquet"
            new_split.to_parquet(output_name)
        else:
            token = args.token or os.environ.get("HF_TOKEN")
            new_split.push_to_hub(args.upload_name, split=split, token=token)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
