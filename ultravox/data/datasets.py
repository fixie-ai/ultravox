import abc
import base64
import dataclasses
import enum
import io
import itertools
import logging
import os
import tempfile
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import datasets
import jinja2
import librosa
import numpy as np
import soundfile as sf
import streaming as mds
import torch
import torch.nn.functional as F
import transformers
from pydantic import BaseModel
from torch.utils import data

from ultravox.data import text_proc

SAMPLE_RATE = 16000

# TODO(juberti): set these in the environment so they don't need to be hard-coded here.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "fixie-training"

# Silence the spurious warnings coming from the MosaicML streaming library.
logging.getLogger("streaming.base.dataset").setLevel(logging.ERROR)


class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"


# Global arguments for voice datasets.
@dataclasses.dataclass
class VoiceDatasetArgs:
    """Global arguments for voice datasets."""

    batch_size: int = 4
    """Batch size for train, eval, or validation."""
    include_audio: bool = True
    """Whether to include audio in the samples."""
    shuffle: bool = False
    """Whether to shuffle the dataset."""
    shuffle_seed: int = 42
    """Seed for shuffling the dataset."""
    max_audio_duration_secs: Optional[float] = None
    """Whether to skip samples with audio longer than this duration."""
    split: DatasetSplit = DatasetSplit.TRAIN
    """Which split of the dataset to use."""

    def __post_init__(self):
        if isinstance(self.split, str):
            self.split = DatasetSplit(self.split.lower())


class DatasetSplitConfig(BaseModel):
    name: str
    """Name of the split"""
    num_samples: int
    """Number of samples in the split"""
    is_validation: bool = False

    def __post_init__(self):
        if self.name == "validation":
            self.is_validation = True


class DatasetConfig(BaseModel):
    path: str = ""
    """Directory of the dataset, or huggingface dataset name; must be set for "generic" datasets. If not set, it is automatically inferred for predefined dataset types."""
    subset: Optional[str] = None
    """Name of the dataset, or huggingface dataset config/subset name"""
    splits: List[DatasetSplit] = []
    """List of splits to use, e.g. [{"name": "train", "num_samples": 1000}, {"name": "validation", "num_samples": 100}]"""
    user_template: str = "<|audio|>"
    """Template for the user's message"""
    user_template_args: Dict[str, str] = {}
    """Optional arguments (e.g., target language) for the user template"""
    assistant_template: str = "{{text}}"
    """Template for the assistant's message"""
    transcript_template: str = "{{text}}"
    """Template for the transcript"""
    audio_field: Optional[str] = "audio"
    """Field in the dataset that contains the audio, use None if the dataset does not contain audio"""
    use_mds: bool = False
    """Set to True to load the dataset from GCP (using MDS) instead of Hugging Face"""
    mds_batch_size: int = 32
    """Batch size for MDS"""

    class Config:
        extra = "forbid"
        # do not allow undefined parameters

    def model_post_init(self, __context: Any) -> None:
        if not self.splits:
            raise ValueError("At least one split must be provided")


@dataclasses.dataclass
class DataCollatorForSeq2SeqWithAudio(transformers.DataCollatorForSeq2Seq):
    # when enabled, the alt_input_ids, alt_attention_mask, and alt_labels fields are used for computing the KL loss in UltravoxModel
    include_alt_fields: bool = False

    def __call__(self, features, *args, **kwargs):
        audio_values = [f.pop("audio_values", None) for f in features]
        if self.include_alt_fields:
            # these fields are hard-coded in the transformer data collator, so they need special handling before calling the super method
            alt_features = [
                {
                    "input_ids": f.pop("alt_input_ids"),
                    "attention_mask": f.pop("alt_attention_mask"),
                    "labels": f.pop("alt_labels"),
                }
                for f in features
            ]
        input_ids_lens = torch.LongTensor([f["input_ids"].shape[-1] for f in features])
        batch = super().__call__(features, *args, **kwargs)
        if self.include_alt_fields:
            alt_batch = super().__call__(alt_features, *args, **kwargs)
            batch["alt_input_ids"] = alt_batch["input_ids"]
            batch["alt_attention_mask"] = alt_batch["attention_mask"]
            batch["alt_labels"] = alt_batch["labels"]

        # Pad the last dimension of all audio_values to the same length, with 0s on the right.
        if audio_values and audio_values[0] is not None:
            max_len = max([x.shape[-1] for x in audio_values])
            batch["audio_values"] = torch.stack(
                [F.pad(x, (0, max_len - x.shape[-1])) for x in audio_values]
            )
            if self.tokenizer.padding_side == "left":
                displacement = batch["input_ids"].shape[-1] - input_ids_lens
                batch["audio_token_start_idx"] += displacement.to(
                    batch["audio_token_start_idx"].device
                )

        return batch


def audio_from_file(path: str) -> np.ndarray:
    """Load audio from a file, converting to float32 PCM @ 16 kHz."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    assert audio.dtype == np.float32
    return audio


def audio_from_buf(buf: bytes) -> np.ndarray:
    """Load audio from a buffer, converting to float32 PCM @ 16 kHz."""
    audio, _ = librosa.load(io.BytesIO(buf), sr=SAMPLE_RATE)
    assert audio.dtype == np.float32
    return audio


def audio_to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert audio to WAV format, 16-bit PCM @ 16 kHz."""
    assert audio.dtype == np.float32
    with io.BytesIO() as buf:
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def audio_to_wav_base64(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Convert audio to a base64-encoded WAV file."""
    return base64.b64encode(audio_to_wav(audio, sample_rate)).decode("utf-8")


def audio_to_data_uri(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Convert audio to a data URI."""
    return f"data:audio/wav;base64,{audio_to_wav_base64(audio, sample_rate)}"


def messages_from_prompt(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


@dataclasses.dataclass
class VoiceSample:
    @staticmethod
    def from_json(data: Dict[str, Any]) -> "VoiceSample":
        """Convert from JSON format; audio is expected as base64ed WAV."""
        bytes = base64.b64decode(data["audio"])
        return VoiceSample(data["messages"], audio_from_buf(bytes))

    @staticmethod
    def from_prompt(prompt: str) -> "VoiceSample":
        """Create a VoiceSample from a prompt only."""
        return VoiceSample(messages_from_prompt(prompt), None)

    @staticmethod
    def from_prompt_and_file(prompt: str, path: str) -> "VoiceSample":
        """Create a VoiceSample from a prompt and an audio file."""
        return VoiceSample(messages_from_prompt(prompt), audio_from_file(path))

    @staticmethod
    def from_prompt_and_buf(prompt: str, buf: bytes) -> "VoiceSample":
        """Create a VoiceSample from a prompt and an encoded audio buffer."""
        return VoiceSample(messages_from_prompt(prompt), audio_from_buf(buf))

    @staticmethod
    def from_prompt_and_raw(
        prompt: str, buf: np.ndarray, sample_rate: int
    ) -> "VoiceSample":
        """Create a VoiceSample from a prompt and raw audio data with sample rate."""
        # Keep in native sample rate; we'll resample later if needed.
        return VoiceSample(messages_from_prompt(prompt), buf, sample_rate)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format; audio is written as base64ed WAV."""
        obj: Dict[str, Any] = {"messages": self.messages}
        if self.audio is not None:
            obj["audio"] = audio_to_wav_base64(self.audio, self.sample_rate)
        return obj

    def __post_init__(self):
        """Ensure audio is float32 PCM."""
        if self.audio is not None:
            if self.audio.dtype == np.float64:
                self.audio = self.audio.astype(np.float32)
            elif self.audio.dtype == np.int16:
                self.audio = self.audio.astype(np.float32) / np.float32(32768.0)
            elif self.audio.dtype == np.int32:
                self.audio = self.audio.astype(np.float32) / np.float32(2147483648.0)
            assert (
                self.audio.dtype == np.float32
            ), f"Unexpected audio dtype: {self.audio.dtype}"
            assert self.audio.ndim == 1, f"Unexpected audio shape: {self.audio.shape}"

    def add_past_messages(self, past_messages: List[Dict[str, str]]):
        self.messages = past_messages + self.messages

    messages: List[Dict[str, str]]
    """List of messages, each with a "role" and "content" field."""
    audio: Optional[np.typing.NDArray[np.float32]] = None
    """Audio data as float32 PCM @ `sample_rate`."""
    sample_rate: int = SAMPLE_RATE
    """Audio sample rate in Hz."""
    audio_transcript: Optional[str] = None
    """For evaluations, the known transcript of the audio."""


def _get_messages(
    *turns: str, sys_prompt: Optional[str] = None, assistant_last: bool = True
) -> List[Dict[str, str]]:
    """
    Convert a list of strings into a list of messages, alternating between user and assistant.
    If `sys_prompt` is set, it is prepended as a system message.
    If `assistant_last` is True, the assistant's message is the last one.
    """
    messages = []

    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    roles = ["user", "assistant"]

    # Make sure the last turn is the assistant's iff assistant_last is True.
    if (len(turns) + assistant_last) % 2 == 0:
        roles = roles[::-1]

    messages += [{"role": roles[i % 2], "content": c} for i, c in enumerate(turns)]

    return messages


class SizedIterableDataset(abc.ABC, data.IterableDataset):
    """
    An interface for an IterableDataset that provides a length method.
    """

    @abc.abstractmethod
    def __len__(self):
        pass


class VoiceDataset(SizedIterableDataset):
    """
    Base class for streaming voice datasets.
    Wraps a Hugging Face dataset or MDS-formatted dataset from GCP.
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__()
        self._args = args
        self._rng = np.random.default_rng(self._args.shuffle_seed)
        if True:  # device_helpers.get_local_rank() == 0:
            logging.info(
                f"Created VoiceDataset with config:\n{self._config.model_dump_json(indent=2)}"
            )

    def _init_dataset(self, dataset: data.Dataset, num_samples: int) -> None:
        self._dataset = dataset
        self._length = num_samples

    def __len__(self):
        return self._length

    def _load_audio_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        shuffle: Optional[bool] = None,
        streaming: bool = True,
    ) -> data.Dataset:
        if shuffle is None:
            shuffle = self._args.shuffle
        if self._args.use_mds:
            gcs_path = path.replace("/", "_")
            if name:
                gcs_path += f"/{name}"
            if split:
                gcs_path += f"/{split}"
            url = f"gs://fixie-datasets/mds/{gcs_path}"
            temp_dir = os.path.join(
                tempfile.gettempdir(), f"mds_{gcs_path.replace('/', '_')}"
            )
            return mds.StreamingDataset(
                remote=url,
                local=temp_dir,
                batch_size=self._args.mds_batch_size,
                shuffle=shuffle,
                shuffle_seed=self._args.shuffle_seed,
            )
        else:
            # HF datasets sometimes fails to download due to network issues, so retry a few times.
            dataset = datasets.load_dataset(
                path,
                name,
                split=split,
                trust_remote_code=True,
                streaming=streaming,
                download_config=datasets.DownloadConfig(max_retries=10),
            )
            if shuffle:
                dataset = dataset.shuffle(seed=self._args.shuffle_seed)
            return dataset

    def __iter__(self):
        actual_length = 0
        for _, row in enumerate(self._dataset):
            sample = self._get_sample(row)
            if sample is None:
                raise ValueError(
                    f"Sample is None in dataset {self._config.alias} for row {row}"
                )

            if self._config.audio_field is not None:
                # If audio_field is set, make sure the sample has audio data.
                if sample.audio is None:
                    raise ValueError(
                        f"Audio field ({self._config.audio_field}) is None in dataset {self._config.alias} for sample {sample}"
                    )
                if sample.audio.shape[-1] == 0:
                    raise ValueError(
                        f"Audio length is 0 in dataset {self._config.alias} for sample {sample}"
                    )
                if (
                    self._args.max_audio_duration_secs is not None
                    and sample.audio.shape[-1] / SAMPLE_RATE
                    > self._args.max_audio_duration_secs
                ):
                    warnings.warn(
                        f"Audio length ({sample.audio.shape[-1] / SAMPLE_RATE}s) exceeds max audio duration ({self._args.max_audio_duration_secs}s) in dataset {self._config.alias}, skipping sample."
                    )
                    continue

            yield sample
            actual_length += 1
            if actual_length == len(self) + 1:
                warnings.warn(
                    f"The presumed length {self._length} has been exceeded for dataset {self._config.alias}. Make sure to update."
                )
        if actual_length != len(self):
            warnings.warn(
                f"Mismatch between presumed length ({self._length}) and actual length ({actual_length}) for dataset {self._config.alias}. Make sure to update."
            )

    @abc.abstractmethod
    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        """
        Converts a row from the dataset into a VoiceSample.
        Returns None if the sample should be skipped.
        """

    def _get_audio(
        self, row: transformers.BatchFeature, column_name: Optional[str] = "audio"
    ) -> np.ndarray:
        if column_name not in self._config.base_audio_columns:
            raise ValueError(
                f"Unknown audio column: {column_name}. This is likely a bug and the audio might not be resampled to {SAMPLE_RATE} Hz."
            )

        # Hugging Face datasets have an Audio object, with array and sampling_rate fields.
        # For MDS, this object is flattened into audio_array and audio_sampling_rate fields.
        if column_name in row:
            audio = row[column_name]["array"]
            sampling_rate = row[column_name]["sampling_rate"]
        elif f"{column_name}_array" in row:
            audio = row[f"{column_name}_array"]
            sampling_rate = row[f"{column_name}_sampling_rate"]
        else:
            raise ValueError("No audio field found in row.")
        assert sampling_rate == SAMPLE_RATE
        return audio

    def _make_sample(
        self,
        messages: List[Dict[str, str]],
        audio: np.ndarray,
        audio_transcript: Optional[str] = None,
    ) -> VoiceSample:
        if not self._args.include_audio:
            return VoiceSample(messages)
        return VoiceSample(messages, audio, audio_transcript=audio_transcript)


class GenericDataset(VoiceDataset):
    def __init__(self, args: VoiceDatasetArgs, config: DatasetConfig) -> None:
        super().__init__(args)
        self._config = config
        split_names = [
            split.name
            for split in config.splits
            if split.is_validation == (self._args.split == DatasetSplit.VALIDATION)
        ]
        dsets = []
        total_samples = 0
        for split_name in split_names:
            ds = self._load_audio_dataset(config.path, config.name, split=split_name)
            ds = ds.cast_column(
                config.audio_field, datasets.Audio(sampling_rate=SAMPLE_RATE)
            )
            dsets.append(ds)
            total_samples += len(ds)
        dataset = datasets.concatenate_datasets(dsets)
        super()._init_dataset(dataset, total_samples)

    def _get_sample(self, row) -> Optional[VoiceSample]:
        try:
            user_content = jinja2.Template(
                self._config.user_template, undefined=jinja2.StrictUndefined
            ).render(
                **row,
                text_proc=text_proc,
                dataset=self,
                **self._config.user_template_args,
            )
            assistant_content = jinja2.Template(
                self._config.assistant_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc, dataset=self)
            transcript = jinja2.Template(
                self._config.transcript_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc, dataset=self)
        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"user_template: {self._config.user_template}")
            print(f"assistant_template: {self._config.assistant_template}")
            print(f"transcript_template: {self._config.transcript_template}")
            print(f"sample keys: {list(row.keys())}")
            raise ValueError(
                "Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e

        return self._make_sample(
            _get_messages(user_content, assistant_content),
            self._get_audio(row, self._config.audio_field),
            audio_transcript=transcript,
        )


# Making EmptyDataset a SizedIterableDataset to be compatible with using epochs during training.
class EmptyDataset(SizedIterableDataset):
    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


DATASET_MAP: Dict[str, Any] = {}


def register_datasets(datasets: Dict):
    for dataset in datasets:
        DATASET_MAP[dataset] = create_dataset(dataset, datasets[dataset])


def create_dataset(
    args: VoiceDatasetArgs, config: DatasetConfig
) -> SizedIterableDataset:
    configs = []
    while True:
        configs.append(config)
        base = config.get("base")
        if not base:
            break
        config = base
    merged_config = configs[-1]
    for config in configs[:-1]:
        merged_config.update(config)
    del merged_config["base"]
    return GenericDataset(args, merged_config)


class StopStrategy(str, Enum):
    FIRST_EXHAUSTED = "FIRST_EXHAUSTED"
    LAST_EXHAUSTED = "LAST_EXHAUSTED"
    NEVER_STOP = "NEVER_STOP"


@dataclasses.dataclass
class DatasetAndWeight:
    dataset: SizedIterableDataset
    weight: float


class InterleaveDataset(SizedIterableDataset):
    """Interleaves multiple IterableDataset objects based on normalized weights."""

    def __init__(
        self,
        datasets: Sequence[DatasetAndWeight],
        stop_strategy: StopStrategy = StopStrategy.LAST_EXHAUSTED,
        seed: Optional[int] = 42,
        static: bool = False,
    ) -> None:
        """
        Args:
            datasets: A list of SizedIterableDataset objects.
            stop_strategy: Strategy for stopping iteration.
            seed: Optional seed for reproducibility.
            static: If true, the datasets are interleaved in a static order with equal weights.
        """
        self._datasets = [dataset for dataset, _ in datasets]
        self._rng = np.random.default_rng(seed)
        self._static = static
        self._stop_strategy = stop_strategy

        weights = [weight for _, weight in datasets]
        total_weight = sum(weights)
        self._normalized_probs = [w / total_weight for w in weights]

    def __iter__(self):
        # If no datasets are provided, return an empty iterator
        if not self._datasets:
            return

        iters = [iter(ds) for ds in self._datasets]
        exhausted = [False] * len(iters)

        if self._static:
            static_iter = itertools.cycle(range(len(self._datasets)))

        while True:
            if self._static:
                iter_index = next(static_iter)
            else:
                iter_index = self._rng.choice(len(iters), p=self._normalized_probs)

            try:
                yield next(iters[iter_index])
            except StopIteration:
                exhausted[iter_index] = True

                # Check if stopping condition is met
                if self._stop_strategy == StopStrategy.FIRST_EXHAUSTED or (
                    self._stop_strategy == StopStrategy.LAST_EXHAUSTED
                    and all(exhausted)
                ):
                    break

                # Recreate the iterator if stopping condition is not met and yield the next sample
                iters[iter_index] = iter(self._datasets[iter_index])
                yield next(iters[iter_index])

    def __len__(self):
        # TODO: Implement the length method for different stop strategies
        return sum(len(ds) for ds in self._datasets)


class Dataproc(SizedIterableDataset):
    """Base class to preprocess a dataset of VoiceSamples."""

    def __init__(self, dataset: SizedIterableDataset) -> None:
        self._dataset = dataset

    @abc.abstractmethod
    def _process(self, sample: VoiceSample) -> Dict[str, Any]:
        pass

    def __iter__(self):
        return (self._process(sample) for sample in self._dataset)

    def __len__(self):
        return len(self._dataset)


class Range(SizedIterableDataset):
    """Limits the number of samples from another dataset."""

    def __init__(
        self,
        dataset: data.IterableDataset,
        num_samples: Optional[int] = None,
        total_samples: Optional[int] = None,
    ) -> None:
        self._dataset = dataset
        self._num_samples = num_samples

        if isinstance(self._dataset, SizedIterableDataset):
            self._estimated_length = len(self._dataset)
        else:
            if total_samples is None:
                raise ValueError(
                    "total_samples must be provided for non-SizedIterableDataset."
                )
            self._estimated_length = total_samples

        if self._num_samples is not None and self._num_samples > self._estimated_length:
            # Issuing a warning here instead of raising an error to accomodate for specific classes of VoiceDataset
            # Once we migrate entirely to GenericVoiceDataset, we can raise an error here.
            warnings.warn("num_samples is greater than total_samples.")

    def __iter__(self):
        for i, sample in enumerate(self._dataset):
            if self._num_samples is not None and i >= self._num_samples:
                break
            yield sample

    def __len__(self):
        return self._length
