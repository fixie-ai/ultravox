import dataclasses
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import simple_parsing
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_fixed

import ultravox.tools.ds_tool.chunked_dataset as chunked_dataset
from ultravox.tools.ds_tool.tasks import dedup_task
from ultravox.tools.ds_tool.tasks import text_gen_task
from ultravox.tools.ds_tool.tasks import timestamp_gen_task
from ultravox.tools.ds_tool.tasks import tts_task


# This script is used to either generate audio samples from text using a TTS model, or to generate text samples using a text generation model.
#   just ds_tool tts -d google/boolq -u fixie-ai/boolq-audio -T {{question}} -a audio --token $HF_WRITE_TOKEN
#   just ds_tool textgen -d fixie-ai/boolq-audio -u fixie-ai/bar -T {{explanation}} -b https://api.fireworks.ai/inference/v1 -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct
#   just ds_tool textgen -d ylacombe/expresso -u fixie-ai/expresso -T {{continuation}} -T @expresso_template.txt
#   just ds_tool textgen --new_column_name continuation --dataset_name openslr/librispeech_asr --dataset_subset clean --dataset_split train.360 \
#        --shuffle --upload_name fixie-ai/librispeech_asr --private --base_url https://api.fireworks.ai/inference/v1 \
#        --api_key $FIREWORKS_API_KEY --token $HF_TOKEN --language_model accounts/fireworks/models/llama-v3-8b-instruct \
#        --template @ultravox/tools/ds_tool/continuation.jinja --max_tokens 64 --num_workers 30 --writer_batch_size 30
#   just ds_tool timestamp -d fixie-ai/common_voice_17_0 -S en --upload_name fixie-ai/cv_ts  \
#        -m english_mfa -T "\"{{text_proc.format_asr_text(sentence)}}\""
#   just ds_tool dedup -d fixie-ai/proper-noun-challenge -T "\"{{user}}\"" -t 0.6 -u fixie-ai/proper-noun-challenge-filtered --check_empty_columns --chunk_split_threshold 1000000
@dataclasses.dataclass
class DatasetToolArgs:
    # HF source dataset parameters
    dataset_name: str = simple_parsing.field(alias="-d")
    dataset_subset: Optional[str] = simple_parsing.field(default=None, alias="-S")
    dataset_split: Optional[str] = simple_parsing.field(default=None, alias="-s")
    dataset_version: Optional[str] = simple_parsing.field(default="main", alias="-v")

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
    num_chunks: int = simple_parsing.field(default=10)
    chunk_split_threshold: int = simple_parsing.field(default=50000)

    # Columns that cannot be null
    check_empty_columns: List[str] = simple_parsing.field(
        default_factory=lambda: ["audio"]
    )

    task: Union[
        tts_task.TtsTask,
        text_gen_task.TextGenerationTask,
        timestamp_gen_task.TimestampGenerationTask,
        dedup_task.DeduplicationTask,
    ] = simple_parsing.subgroups(
        {
            "tts": tts_task.TtsTask,
            "textgen": text_gen_task.TextGenerationTask,
            "timestamp": timestamp_gen_task.TimestampGenerationTask,
            "dedup": dedup_task.DeduplicationTask,
        },  # type: ignore
        default_factory=tts_task.TtsTask,
        positional=True,
    )

    def __post_init__(self):
        if not self.dataset_subset:
            self.dataset_subset = "default"
        if not self.upload_subset and self.dataset_subset:
            self.upload_subset = self.dataset_subset
        if self.dataset_split and not self.upload_split:
            self.upload_split = self.dataset_split

        if self.upload_name == self.dataset_name:
            raise ValueError(
                "Updating datasets in-place is not well-supported and hence frowned upon."
            )


class DatasetChunkProcessor:
    args: DatasetToolArgs
    cache_dir: str = ".cache/ds_tool/processed_datasets"
    chunks_not_uploaded: List[Tuple[int, int]] = []

    def __init__(self, args: DatasetToolArgs):
        self.args = args

    def process_and_upload_split_rescursive(
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
            total_chunks = self.args.num_chunks
            chunk_size = math.ceil(original_chunk_size / total_chunks)
        failed_chunk_ranges = []
        print(
            f"Processing and uploading {total_chunks} chunks for range [{start_index}, {end_index}) with chunk size {chunk_size}"
        )
        for chunk_start in range(start_index, end_index, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_index)

            ds_chunk = ds_split.select(range(chunk_start, chunk_end))
            ds_chunk_name = f"chunk-range-{chunk_start:09d}-{chunk_end:09d}"
            ds_chunk_hub_path = os.path.join(
                str(self.args.upload_subset), split_name, ds_chunk_name
            )
            ds_chunk_cache_path = os.path.join(
                self.cache_dir,
                self.args.dataset_name.replace("/", "__"),
                str(self.args.upload_subset),
                split_name,
                ds_chunk_name + ".parquet",
            )
            try:
                if os.path.exists(ds_chunk_cache_path):
                    print(
                        f"Skipping chunk {ds_chunk_name} as it has already been processed and uploaded."
                    )
                else:
                    print(f"Processing chunk {ds_chunk_name}")
                    ds_chunk_processed = self._process(ds_chunk)
                    print(
                        "Finished processing chunk with length", len(ds_chunk_processed)
                    )
                    if len(ds_chunk_processed) > 0:
                        # Note: The caching is after the upload to avoid caching failed upload chunks.
                        # Saved chunks indicate they have been uploaded to HF.
                        self._upload(ds_chunk_processed, ds_chunk_hub_path, split_name)
                        ds_chunk_processed.to_parquet(ds_chunk_cache_path)
                    else:
                        print(f"Chunk {ds_chunk_name} has 0 samples. Not uploading.")

            except Exception as e:
                # If the error is unsupported operand type(s) for -=: 'NoneType' and 'float',
                # then the huggingface README needs to be updated to have the
                # download_size, and dataset_size fields present under dataset_info (could be initalized to 0)
                print(f"Failed to upload chunk {ds_chunk_name}: {e}. Retrying later.")
                print(traceback.format_exc())
                if total_chunks == 1:
                    print(
                        f"Finished processing and uploading 0/1 chunks for range [{start_index}, {end_index})"
                    )
                    self.chunks_not_uploaded.append((start_index, end_index))
                    return None
                failed_chunk_ranges.append((chunk_start, chunk_end))
        successful_chunks = self.args.num_chunks - len(failed_chunk_ranges)
        print(
            f"Finished processing and uploading {successful_chunks}/{self.args.num_chunks} chunks for range [{start_index}, {end_index})"
        )
        if len(failed_chunk_ranges) > 0:
            for start, end in failed_chunk_ranges:
                print(f"Retrying failed chunk range [{start}, {end})")
                self.process_and_upload_split_rescursive(
                    split_name, ds_split, start, end
                )

        print(f"Could not upload chunks: {self.chunks_not_uploaded}")
        print(f"Finished processing and uploading all chunks for split {split_name}.")

    def _process(self, ds_chunk: datasets.Dataset) -> datasets.Dataset:
        ds_mapped = self.args.task.map_split(
            ds_chunk,
            self.args.num_workers,
            self.args.writer_batch_size,
            self.args.exclude_fields,
        )

        check_empty_columns = self.args.check_empty_columns
        if len(check_empty_columns) > 0:
            return ds_mapped.filter(
                lambda sample: all(
                    sample[column] is not None for column in check_empty_columns
                ),
                num_proc=self.args.num_workers,
                writer_batch_size=self.args.writer_batch_size,
            )
        else:
            return ds_mapped

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3))
    def _upload(self, ds_chunk_processed: datasets.Dataset, data_dir: str, split_name):
        print(f"Uploading chunk to hub: {data_dir}")
        ds_split_chunked: chunked_dataset.ChunkedDataset = (
            chunked_dataset.convert_to_chunked_dataset(ds_chunk_processed)
        )

        hub_args: Dict[str, Any] = {
            "config_name": self.args.upload_subset,
            "token": self.args.token or os.environ.get("HF_TOKEN"),
            "private": self.args.private,
            "data_dir": data_dir,
            "num_shards": self.args.num_shards,
            "split": split_name,
        }
        assert isinstance(self.args.upload_name, str)
        ds_split_chunked.push_to_hub(self.args.upload_name, **hub_args)


def main(args: DatasetToolArgs):
    ds_name = args.dataset_name
    print(f'Loading dataset "{ds_name}" for task {args.task}')
    download_config = datasets.DownloadConfig(num_proc=args.num_workers, max_retries=2)
    data_dict: datasets.DatasetDict = datasets.load_dataset(
        ds_name,
        args.dataset_subset,
        split=args.dataset_split,
        download_config=download_config,
        revision=args.dataset_version,
    )

    if isinstance(data_dict, datasets.Dataset):
        data_dict = datasets.DatasetDict({args.upload_split: data_dict})

    if len(data_dict) > 1 and args.upload_split:
        raise ValueError("Cannot upload multiple splits to a single split")

    ds_chunk_proc = DatasetChunkProcessor(args)

    for split_name, ds_split in data_dict.items():
        print(
            f"Processing dataset: {ds_name}, subset {args.dataset_subset}, split {split_name}, containing {len(ds_split)} samples"
        )
        if args.shuffle:
            ds_split = ds_split.shuffle(seed=args.shuffle_seed)
        if args.num_samples:
            ds_split = ds_split.select(range(args.num_samples))

        ds_chunk_proc.process_and_upload_split_rescursive(
            split_name, ds_split, 0, len(ds_split)
        )


if __name__ == "__main__":
    main(simple_parsing.parse(DatasetToolArgs))
