import dataclasses
import glob
import json
import math
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import datasets
import jinja2
import librosa
import openai
import simple_parsing
import soundfile as sf
import yaml
from praatio import textgrid
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_fixed

import ultravox.tools.ds_tool.chunked_dataset as chunked_dataset
from ultravox.data import text_proc
from ultravox.tools.ds_tool import caching
from ultravox.tools.ds_tool import tts

tts_client: caching.CachingTtsWrapper
chat_client: caching.CachingChatWrapper

MFA_ENV_NAME = "aligner"


def apply_jinja_template(
    template: str, sample: Dict[str, Any], exclude_fields: Optional[Set[str]] = None
):
    """
    Apply a Jinja template to a sample, rendering it into text.
    Jinja template allows for added flexibility as template can include variables and functions.

    Args:
        template: The Jinja template to apply. It can include variables, functions, and control structures.
            Example:
                {{ text }}
                {{ text_proc.format_asr_text(text) }}
        sample: The sample to apply the template to.
        exclude_fields: Fields to exclude from the sample before rendering the template, to avoid loading large fields into memory.
    """
    if exclude_fields:
        # Filter out big fields like audio before the sample is passed into the jinja template
        # otherwise it can lead to unnecessary memory usage.
        sample = {k: sample[k] for k in sample.keys() if k not in exclude_fields}

    try:
        return jinja2.Template(template, undefined=jinja2.StrictUndefined).render(
            **sample, json_dump=json.dumps, text_proc=text_proc
        )
    except jinja2.TemplateError as e:
        print(f"Error rendering template: {e}")
        print(f"template: {template}")
        print(f"sample keys: {list(sample.keys())}, excluded keys: {exclude_fields}")
        raise ValueError(
            f"Template rendering failed. Make sure all keys in the template exist in the sample."
        ) from e


@dataclasses.dataclass
class TtsTask:
    # Jinja template for the text that needs to be converted to audio
    template: str = simple_parsing.field(alias="-T")
    implementation: str = simple_parsing.field(default="azure", alias="-i")
    json_mode: bool = simple_parsing.field(default=False, alias="-j")
    audio_column_name: Optional[str] = simple_parsing.field(default=None, alias="-a")
    voice: Optional[str] = simple_parsing.field(default=None, alias="-V")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")

    def __post_init__(self):
        # The TTS client is separate from the task to avoid pickling issues when multiprocessing.
        global tts_client
        if self.audio_column_name is None:
            self.audio_column_name = f"{self.column_name}_audio"
        tts_client = caching.CachingTtsWrapper(
            tts.create_client(self.implementation, self.sample_rate),
            implementation=self.implementation,
        )

        if self.template.startswith("@"):
            with open(self.template[1:], "r") as template_file:
                self.template = template_file.read()

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: List[str],
    ) -> datasets.Dataset:
        print(f'TTS mapping "{self.template}" to "{self.audio_column_name}"...')
        ds_split = ds_split.map(
            self._map_sample,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"exclude_fields": exclude_fields},
        )
        column_type = datasets.Audio(sampling_rate=self.sample_rate)
        if self.json_mode and isinstance(
            ds_split.features[self.audio_column_name], datasets.Sequence
        ):
            column_type = datasets.Sequence(column_type)
        return ds_split.cast_column(self.audio_column_name, column_type)

    def _map_sample(self, sample, exclude_fields: Set[str]):
        # using a Jinja template for some added flexibility, template can include variables and functions
        # e.g., {{ text }} or {{ text_proc.format_asr_text(text) }}
        text_or_texts = apply_jinja_template(self.template, sample, exclude_fields)

        if self.json_mode:
            text_or_texts = yaml.safe_load(text_or_texts)
            assert isinstance(text_or_texts, list)
            assert all(isinstance(turn, str) for turn in text_or_texts)

        sample[self.audio_column_name] = tts_client.tts(text_or_texts, self.voice)
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
        chat_client = caching.CachingChatWrapper(
            openai.Client(base_url=self.base_url, api_key=self.api_key),
            unique_id=f"{self.base_url}__{self.language_model}",
        )
        if self.template.startswith("@"):
            with open(self.template[1:], "r") as template_file:
                self.template = template_file.read()

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: List[str],
    ) -> datasets.Dataset:
        print(f'Generating "{self.new_column_name}" with template:\n{self.template}')
        ds_mapped = ds_split.map(
            lambda sample: self._map_sample(sample, set(exclude_fields)),
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
        )

        # Filter out samples where new_column_name is None
        return ds_mapped.filter(
            lambda sample: sample[self.new_column_name] is not None,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
        )

    def _map_sample(self, sample, exclude_fields):
        # using a Jinja template for some added flexibility, template can include variables and functions
        # e.g., {{ text }} or {{ text_proc.format_asr_text(text) }}
        try:
            rendered = apply_jinja_template(self.template, sample, exclude_fields)
        except text_proc.FormatASRError as e:
            print(f"Format ASR Error {e}. Filtering out sample.")
            sample[self.new_column_name] = None
            return sample

        if self.json_mode:
            turns = yaml.safe_load(rendered)
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


@dataclasses.dataclass
class TimestampGenerationTask:
    """
    This task is used to generate timestamps for the text transcription.
    It uses the Montreal Forced Aligner (MFA) to align the text with the audio. The result is a
    list of timestamps for each word in the text transcription. The timestamps are stored in a new
    column, in a list of dict format:
        [ {"start": float in seconds, "end": float in seconds, "text": first word str}, ... ]
    """

    # Jinja template for the text transcription that needs to be aligned
    template: str = simple_parsing.field(alias="-T")
    # The accoustic model to use for MFA alignment.
    # Make sure the dictionary and acoustic model are installed. See just install_mfa for an example (English).
    # Model index: https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
    # For many languages there exists a {language}_mfa model that you can use, e.g. "english_mfa"
    mfa_acoustic_model: str = simple_parsing.field(alias="-m")
    # The dictionary to use for MFA alignment. Defaults to the same name as the acoustic model.
    mfa_dictionary: Optional[str] = simple_parsing.field(default=None, alias="-d")
    audio_column_name: str = simple_parsing.field(default="audio", alias="-a")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")
    # The column name to store the timestamps in
    timestamp_column_name: str = simple_parsing.field(default="timestamps", alias="-ts")
    aligned_ratio_check: float = simple_parsing.field(default=0.95, alias="-ar")

    def __post_init__(self):
        if self.mfa_dictionary is None:
            self.mfa_dictionary = self.mfa_acoustic_model

        try:
            # Make sure the MFA environment is installed
            subprocess.run(["conda", "run", "-n", MFA_ENV_NAME, "echo"], check=True)
        except subprocess.CalledProcessError:
            raise Exception(
                "Please install the MFA environment using `just install_mfa` first."
            )

        if self.template.startswith("@"):
            with open(self.template[1:], "r") as template_file:
                self.template = template_file.read()

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: List[str],
    ) -> datasets.Dataset:
        # 0. create a temp directory to store the audio and text files
        # The files will be deleted when the with block ends or when an exception is raised
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. copy all audio-text pairs into the temp directory
            ds_split.map(
                self._store_sample_as_files,
                num_proc=num_proc,
                fn_kwargs={"exclude_fields": set(exclude_fields), "temp_dir": temp_dir},
            )

            count_wavs = len(glob.glob(os.path.join(temp_dir, "*.wav")))
            assert count_wavs == len(
                ds_split
            ), "Not all samples were stored as files. The id is likely not unique."

            # 2. run the alignment
            self._run_alignment(temp_dir, num_proc=num_proc)

            # 3. retrieve the timestamps
            ds_mapped = ds_split.map(
                self._retrieve_timestamps,
                num_proc=num_proc,
                writer_batch_size=writer_batch_size,
                fn_kwargs={"temp_dir": temp_dir},
            )

            # 4. filter out samples without timestamps (should be a small number)
            ds_mapped = ds_mapped.filter(
                lambda sample: sample[self.timestamp_column_name] is not None,
                num_proc=num_proc,
                writer_batch_size=writer_batch_size,
            )

            # 5. make sure most samples have timestamps
            if len(ds_split) * self.aligned_ratio_check > len(ds_mapped):
                raise Exception(
                    f"Found too many samples without timestamps: {len(ds_mapped)}/{len(ds_split)} aligned."
                )

        return ds_mapped

    def _retrieve_timestamps(self, sample, temp_dir: str):
        # find the timestamps for the audio and populate the timestamps column
        sample_id = self.get_id(sample)
        text_path = os.path.join(temp_dir, f"{sample_id}.TextGrid")
        if not os.path.exists(text_path):
            sample[self.timestamp_column_name] = None
            return sample

        tg = textgrid.openTextgrid(text_path, False)
        timestamps = tg.getTier("words").entries
        sample[self.timestamp_column_name] = [
            {"start": entry.start, "end": entry.end, "text": entry.label}
            for entry in timestamps
        ]
        return sample

    @staticmethod
    def get_id(sample):
        for key in ["id", "segment_id"]:
            if key in sample and isinstance(sample[key], str):
                return str(sample[key])
        for key in ["file", "path", "audio_file"]:
            if key in sample and isinstance(sample[key], str):
                return Path(sample[key]).stem
        raise ValueError("Could not find an ID in the sample")

    def _store_sample_as_files(self, sample, temp_dir: str, exclude_fields: Set[str]):
        sample_id = self.get_id(sample)
        audio_path = os.path.join(temp_dir, f"{sample_id}.wav")
        with open(audio_path, "wb") as f:
            audio = sample[self.audio_column_name]
            if audio["sampling_rate"] != self.sample_rate:
                audio["array"] = librosa.resample(
                    audio["array"],
                    orig_sr=audio["sampling_rate"],
                    target_sr=self.sample_rate,
                )
            sf.write(f, audio["array"], 16000, format="WAV", subtype="PCM_16")

        text_path = os.path.join(temp_dir, f"{sample_id}.txt")
        text = apply_jinja_template(self.template, sample, exclude_fields)
        with open(text_path, "w") as f:
            f.write(text)

    def _run_alignment(self, temp_dir: str, num_proc: int = 16) -> None:
        subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                MFA_ENV_NAME,
                "mfa",
                "align",
                "--clean",
                "--single_speaker",
                "--use_mp",
                "-j",
                str(num_proc),
                temp_dir,
                self.mfa_acoustic_model,
                str(self.mfa_dictionary),
                temp_dir,
            ],
            check=True,
        )


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

    task: Union[TtsTask, TextGenerationTask, TimestampGenerationTask] = (
        simple_parsing.subgroups(
            {"tts": TtsTask, "textgen": TextGenerationTask, "timestamp": TimestampGenerationTask},  # type: ignore
            default_factory=TtsTask,
            positional=True,
        )
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
