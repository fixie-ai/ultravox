# based upon https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/parallel_dataset_conversion.html

import dataclasses
import json
import logging
import multiprocessing
import os
import shutil
from typing import Any, Iterator, Optional

import datasets  # HuggingFace
import gcsfs
import numpy as np
import simple_parsing
import streaming
from fsspec import callbacks

"""
Commands to convert existing datasets to MDS:
just mds -d librispeech_asr -s train.clean.100 -u -v
just mds -d librispeech_asr -s train.clean.360 -u -v
just mds -d librispeech_asr -s train.other.500 -u -v
just mds -d facebook/voxpopuli -S en -s train -u -v
just mds -d MLCommons/peoples_speech -s train -u -v
just mds -d mozilla-foundation/common_voice_16_1 -S en -s train -u -v
just mds -d speechcolab/gigaspeech -S xl -s train -u -v
just mds -d fixie-ai/boolq-audio -s train -u -v
"""


@dataclasses.dataclass
class MdsArgs:
    dataset_name: str = simple_parsing.field(alias="-d")
    dataset_subset: Optional[str] = simple_parsing.field(default=None, alias="-S")
    dataset_split: Optional[str] = simple_parsing.field(default=None, alias="-s")
    output_dir: str = "./mds_output"
    num_procs: int = 8
    num_groups: int = 8
    gcp_project: str = "fixie-training"
    gcp_bucket: str = "fixie-datasets"
    gcp_path: str = "mds"
    upload: bool = simple_parsing.field(default=False, alias="-u")
    verbose: bool = simple_parsing.field(default=False, alias="-v")


class MdsConverter:
    @dataclasses.dataclass
    class _ProcessArgs:
        columns: dict
        out: str
        start_idx: int
        end_idx: int

    def __init__(self, args: MdsArgs):
        self._args = args
        if self._args.verbose:
            logging.basicConfig(level=logging.INFO)

        print("Loading dataset...")
        self._dataset = datasets.load_dataset(
            self._args.dataset_name,
            self._args.dataset_subset,
            split=self._args.dataset_split,
            trust_remote_code=True,
        )
        logging.info(
            f"Loaded {self._dataset}, subset={self._args.dataset_subset} split={self._args.dataset_split}"
        )

    def run(self) -> None:
        path = self._args.dataset_name.replace("/", "_")
        if self._args.dataset_subset:
            path = os.path.join(path, self._args.dataset_subset)
        if self._args.dataset_split:
            path = os.path.join(path, self._args.dataset_split)
        self.convert(path)
        if self._args.upload:
            self.upload(path)

    def convert(self, path: str) -> None:
        data_dir = os.path.join(self._args.output_dir, path)

        # Clean out any previous conversion.
        if os.path.exists(data_dir):  # and self._force_deletion:
            shutil.rmtree(data_dir)

        # Download the dataset in parallel and write via a single writer.
        columns = self._map_columns(self._dataset.features)
        tasks = self._create_tasks(columns, data_dir, self._args.num_groups)
        n = 0

        print(
            f"Starting conversion, groups={self._args.num_groups}, procs={self._args.num_procs}"
        )
        with multiprocessing.Pool(
            initializer=self._init_worker, processes=self._args.num_procs
        ) as pool:
            for count in pool.imap(self._convert_worker, tasks):
                n += count
        print("Merging indexes...")

        streaming.base.util.merge_index(data_dir, keep_local=True)
        print(f"Conversion completed, samples={n}, path={data_dir}")

    def _map_columns(self, features: dict) -> dict:
        def map_dtype(dtype: str) -> str:
            if dtype == "bool":
                return "int"
            elif dtype == "string":
                return "str"
            return dtype

        # Rewrite type names to match MDS.
        columns = {k: map_dtype(v.dtype) for k, v in features.items()}
        # Remap any audio structure to an array and a sample rate.
        if "audio" in columns:
            del columns["audio"]
            columns["audio_array"] = "ndarray:float32"
            columns["audio_sampling_rate"] = "int"
        return columns

    def _create_tasks(
        self, columns: dict, out_root: str, num_groups: int
    ) -> Iterator[_ProcessArgs]:
        for group in range(num_groups):
            sub_out_root = os.path.join(out_root, str(group))
            num_samples = len(self._dataset)
            batch_size = num_samples // num_groups + 1
            start_idx = group * batch_size
            end_idx = min(start_idx + batch_size, num_samples) - 1
            yield self._ProcessArgs(columns, sub_out_root, start_idx, end_idx)

    def _init_worker(self) -> None:
        if self._args.verbose:
            logging.basicConfig(level=logging.INFO)

    def _convert_worker(self, task_args: _ProcessArgs) -> int:
        n = 0
        with streaming.MDSWriter(
            out=task_args.out, columns=task_args.columns
        ) as writer:
            for sample in self._process_batch(task_args.start_idx, task_args.end_idx):
                writer.write(sample)
                n += 1
                if task_args.start_idx == 0 and n % 1000 == 0:
                    logging.info(f"Processed {n * self._args.num_groups} samples...")
        return n

    def _process_batch(self, start_idx: int, end_idx: int) -> Any:
        for i in range(start_idx, end_idx + 1):
            row = self._dataset[i]
            audio = row["audio"]
            del row["audio"]
            row["audio_array"] = audio["array"].astype(np.float32)
            row["audio_sampling_rate"] = audio["sampling_rate"]
            yield row

    def upload(self, path: str) -> None:
        data_dir = os.path.join(self._args.output_dir, path)
        token = json.load(open("service_account.json"))
        storage_options = {"project": self._args.gcp_project, "token": token}
        fs = gcsfs.GCSFileSystem(**storage_options)
        uri = f"gcs://{self._args.gcp_bucket}/{self._args.gcp_path}/{path}"
        callback = callbacks.TqdmCallback(tqdm_kwargs={"desc": "Uploading files"})
        fs.upload(
            data_dir,
            uri,
            recursive=True,
            storage_options=storage_options,
            callback=callback,
        )


def main(args: MdsArgs):
    converter = MdsConverter(args)
    converter.run()


if __name__ == "__main__":
    main(simple_parsing.parse(MdsArgs))
