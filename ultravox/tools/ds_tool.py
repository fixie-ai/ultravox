import datasets
from ultravox.tools import tts
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", "-d", type=str, required=True)
parser.add_argument("--column-name", "-c", type=str, required=True)
parser.add_argument("--audio-column-name", "-a", type=str)
parser.add_argument("--output-name", "-o", type=str)


def main(args: argparse.Namespace):
    ds_name = args.dataset_name
    column_name = args.column_name
    audio_column_name = args.audio_column_name or f"{column_name}_audio"
    ds = datasets.load_dataset(
        ds_name, split="train", streaming=True, trust_remote_code=True
    )
    tts_client = tts.AzureTts()

    def _tts_column(example, idx: int):
        audio = tts_client.tts(example[column_name])
        example[audio_column_name] = audio
        if "idx" not in example:
            example["idx"] = idx
        return example

    new_ds = ds.map(_tts_column, with_indices=True, batched=True)
    output_name = args.output_name or f"{ds_name}-audio"
    new_ds.to_parquet(f"{output_name}.parquet")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
