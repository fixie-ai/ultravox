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
from typing import Any, Dict, List, Optional, Sequence

import datasets as hf_datasets
import jinja2
import librosa
import numpy as np
import soundfile as sf
import streaming as mds
import torch
import torch.nn.functional as F
import transformers
from simple_parsing import helpers
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
    TEST = "test"
    VALIDATION = "validation"


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


@dataclasses.dataclass
class DatasetSplitConfig(helpers.Serializable):
    name: str
    """Name of the split."""
    num_samples: int
    """Number of samples in the split"""
    split_type: DatasetSplit = DatasetSplit.TRAIN
    """Type of split, i.e., train or validation."""

    def __post_init__(self):
        """Automatically set is_validation if it's a validation split."""
        if self.name == "test":
            self.split_type = DatasetSplit.TEST
        elif self.name == "validation":
            self.split_type = DatasetSplit.VALIDATION


@dataclasses.dataclass
class DatasetConfig(helpers.Serializable):
    name: str
    """Name of the dataset."""
    base: Optional[str] = None
    """Base dataset config to inherit from."""
    path: Optional[str] = None
    """Directory of the dataset, or huggingface dataset name; must be set for "generic" datasets. If not set, it is automatically inferred for predefined dataset types."""
    subset: Optional[str] = None
    """Name of the dataset, or huggingface dataset config/subset name."""
    splits: Optional[List[DatasetSplitConfig]] = None
    """List of splits to use, e.g. [{"name": "train", "num_samples": 1000}, {"name": "validation", "num_samples": 100}]."""
    user_template: Optional[str] = None
    """Template for the user message."""
    user_template_args: Optional[Dict[str, str]] = None
    """Optional arguments (e.g., target language) for the user template."""
    assistant_template: Optional[str] = None
    """Template for the assistant message."""
    transcript_template: Optional[str] = None
    """Template for the transcript."""
    audio_field: Optional[str] = None
    """Field in the dataset that contains the audio, use None if the dataset does not contain audio."""
    use_mds: Optional[bool] = None
    """Set to True to load the dataset from GCP (using MDS) instead of Hugging Face."""
    mds_batch_size: Optional[int] = None
    """Batch size for the dataset when using MDS."""

    def __post_init__(self):
        """Set defaults only if this is a root config, so that said defaults in a subclass don't act as overrides."""
        DEFAULTS = {
            "splits": [],
            "user_template": "<|audio|>",
            "user_template_args": {},
            "assistant_template": "{{text}}",
            "transcript_template": "{{text}}",
            "audio_field": "audio",
            "use_mds": False,
            "mds_batch_size": 32,
        }
        if self.base is None:
            for attr, default_value in DEFAULTS.items():
                if getattr(self, attr) is None:
                    setattr(self, attr, default_value)


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

    def _init_dataset(self, dataset: data.Dataset, num_samples: int) -> None:
        self._dataset = dataset
        self._length = num_samples

    def __len__(self):
        return self._length

    def _load_hf_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        streaming: bool = True,
        audio_field: Optional[str] = None,
    ) -> data.Dataset:
        # HF datasets sometimes fails to download due to network issues, so retry a few times.
        dataset = hf_datasets.load_dataset(
            path,
            name,
            split=split,
            trust_remote_code=True,
            streaming=streaming,
            download_config=hf_datasets.DownloadConfig(max_retries=10),
        )
        if audio_field is not None:
            dataset = dataset.cast_column(
                audio_field, hf_datasets.Audio(sampling_rate=SAMPLE_RATE)
            )
        if self._args.shuffle:
            dataset = dataset.shuffle(seed=self._args.shuffle_seed)
        return dataset

    def _load_mds_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        batch_size: int = 1,
    ) -> data.Dataset:
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
            batch_size=batch_size,
            shuffle=self._args.shuffle,
            shuffle_seed=self._args.shuffle_seed,
        )

    def __iter__(self):
        actual_length = 0
        for _, row in enumerate(self._dataset):
            sample = self._get_sample(row)
            if sample is None:
                raise ValueError(
                    f"Sample is None in dataset {self._config.alias} for row {row}"
                )

            if self._args.include_audio:
                # If audio_field is set, make sure the sample has audio data.
                if sample.audio is None:
                    raise ValueError(f"Audio is None for sample {sample}")
                if sample.audio.shape[-1] == 0:
                    raise ValueError(f"Audio length is 0 for sample {sample}")
                if (
                    self._args.max_audio_duration_secs is not None
                    and sample.audio.shape[-1] / SAMPLE_RATE
                    > self._args.max_audio_duration_secs
                ):
                    duration = sample.audio.shape[-1] / SAMPLE_RATE
                    warnings.warn(
                        f"Audio length ({duration}s) exceeds max audio duration ({self._args.max_audio_duration_secs}s), skipping sample."
                    )
                    continue

            yield sample
            actual_length += 1
            if actual_length == len(self) + 1:
                warnings.warn(
                    f"The presumed length {self._length} has been exceeded for split {self._dataset.split}. Make sure to update."
                )
        if actual_length != len(self):
            warnings.warn(
                f"Mismatch between presumed length ({self._length}) and actual length ({actual_length}) for split {self._dataset.split}. Make sure to update."
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

    def _make_messages(
        self, user_content: str, assistant_content: str
    ) -> List[Dict[str, str]]:
        return _get_messages(user_content, assistant_content)

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
        assert config.splits is not None
        assert config.path is not None
        assert config.mds_batch_size is not None
        super().__init__(args)
        self._config = config
        dsets = []
        total_samples = 0
        for split in config.splits:
            if split.split_type == self._args.split:
                if not config.use_mds:
                    ds = self._load_hf_dataset(
                        config.path,
                        config.subset,
                        split=split.name,
                        audio_field=config.audio_field,
                    )
                else:
                    ds = self._load_mds_dataset(
                        config.path,
                        name=config.subset,
                        split=split.name,
                        batch_size=config.mds_batch_size,
                    )
                dsets.append(ds)
                total_samples += split.num_samples
        dataset = ds if len(dsets) == 1 else hf_datasets.concatenate_datasets(dsets)
        super()._init_dataset(dataset, total_samples)

    def _get_sample(self, row) -> Optional[VoiceSample]:
        assert self._config.user_template is not None
        assert self._config.user_template_args is not None
        assert self._config.assistant_template is not None
        assert self._config.transcript_template is not None
        try:
            user_content = jinja2.Template(
                self._config.user_template, undefined=jinja2.StrictUndefined
            ).render(
                **row,
                text_proc=text_proc,
                dataset=self,
                include_audio=self._args.include_audio,
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


class EmptyDataset(SizedIterableDataset):
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __iter__(self):
        return iter([])

    def __len__(self):
        return self._length


class StopStrategy(str, enum.Enum):
    FIRST_EXHAUSTED = "FIRST_EXHAUSTED"
    LAST_EXHAUSTED = "LAST_EXHAUSTED"
    NEVER_STOP = "NEVER_STOP"


class InterleaveDataset(SizedIterableDataset):
    """Interleaves multiple IterableDataset objects based on normalized weights."""

    def __init__(
        self,
        datasets: Sequence[SizedIterableDataset],
        weights: Optional[Sequence[float]] = None,
        stop_strategy: StopStrategy = StopStrategy.LAST_EXHAUSTED,
        seed: Optional[int] = 42,
        static: bool = False,
    ) -> None:
        """
        Args:
            datasets: A list of SizedIterableDataset objects.
            weights: A list of weights for each dataset.
            stop_strategy: Strategy for stopping iteration.
            seed: Optional seed for reproducibility.
            static: If true, the datasets are interleaved in a static order with equal weights.
        """
        self._datasets = datasets
        self._rng = np.random.default_rng(seed)
        self._static = static
        self._stop_strategy = stop_strategy

        if weights is None:
            weights = [1.0] * len(datasets)
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
        self, dataset: SizedIterableDataset, num_samples: Optional[int] = None
    ) -> None:
        self._dataset = dataset
        self._length = num_samples or len(dataset)
        if self._length > len(dataset):
            raise ValueError("num_samples exceeds dataset length.")

    def __iter__(self):
        for i, sample in enumerate(self._dataset):
            if i >= self._length:
                break
            yield sample

    def __len__(self):
        return self._length


CONTINUATION_USER_TEMPLATE = (
    "Continue the following text using less than 50 words:\n\n<|audio|>"
)
CONTINUATION_ASSISTANT_TEMPLATE = "{{continuation}}"
TRANSCRIPTION_USER_TEMPLATE = "Transcribe\n<|audio|>"

BOOLQ_CONFIG = DatasetConfig(
    name="boolq",
    path="fixie-ai/boolq-audio",
    splits=[
        DatasetSplitConfig(name="train", num_samples=10000),
        DatasetSplitConfig(name="validation", num_samples=1000),
    ],
    user_template="{{passage}}\n\n{{'<|audio|>' if include_audio else question}}",
    assistant_template="{{'True' if answer else 'False'}}",
    transcript_template="{{question}}",
)

CV_BASE_CONFIG = DatasetConfig(
    name="commonvoice",
    path="fixie-ai/common_voice_17_0",
    assistant_template="{{sentence}}",
    transcript_template="{{sentence}}",
)

CV_EN_CONFIG = DatasetConfig(
    name="commonvoice-en",
    base="commonvoice",
    subset="en",
    splits=[DatasetSplitConfig(name="train", num_samples=1_101_170)],
)

CV_AR_CONFIG = DatasetConfig(
    name="commonvoice-ar",
    base="commonvoice",
    subset="ar",
    splits=[DatasetSplitConfig(name="train", num_samples=28_369)],
)

CV_DE_CONFIG = DatasetConfig(
    name="commonvoice-de",
    base="commonvoice",
    subset="de",
    splits=[DatasetSplitConfig(name="train", num_samples=589_100)],
)

CV_ES_CONFIG = DatasetConfig(
    name="commonvoice-es",
    base="commonvoice",
    subset="es",
    splits=[DatasetSplitConfig(name="train", num_samples=336_846)],
)

CV_FR_CONFIG = DatasetConfig(
    name="commonvoice-fr",
    base="commonvoice",
    subset="fr",
    splits=[DatasetSplitConfig(name="train", num_samples=558_054)],
)

CV_IT_CONFIG = DatasetConfig(
    name="commonvoice-it",
    base="commonvoice",
    subset="it",
    splits=[DatasetSplitConfig(name="train", num_samples=169_771)],
)

CV_JA_CONFIG = DatasetConfig(
    name="commonvoice-ja",
    base="commonvoice",
    subset="ja",
    splits=[DatasetSplitConfig(name="train", num_samples=10_039)],
)

CV_PT_CONFIG = DatasetConfig(
    name="commonvoice-pt",
    base="commonvoice",
    subset="pt",
    splits=[DatasetSplitConfig(name="train", num_samples=21_968)],
)

CV_RU_CONFIG = DatasetConfig(
    name="commonvoice-ru",
    base="commonvoice",
    subset="ru",
    splits=[DatasetSplitConfig(name="train", num_samples=26_377)],
)

GS_XL_CONFIG = DatasetConfig(
    name="gigaspeech",
    path="speechcolab/gigaspeech",
    subset="xl",
    splits=[
        DatasetSplitConfig(name="train", num_samples=1_000_000),
        DatasetSplitConfig(name="validation", num_samples=10_000),
    ],
    assistant_template="{{text_proc.format_asr_text(text)}}",
    transcript_template="{{text_proc.format_asr_text(text)}}",
)

LS_BASE_CONFIG = DatasetConfig(
    name="librispeech",
    path="fixie-ai/librispeech_asr",
    assistant_template="{{text_proc.format_asr_text(text)}}",
    transcript_template="{{text_proc.format_asr_text(text)}}",
)

LS_CLEAN_CONFIG = DatasetConfig(
    name="librispeech-clean",
    base="librispeech",
    subset="clean",
    splits=[
        DatasetSplitConfig(name="train.100", num_samples=28_539),
        DatasetSplitConfig(name="train.360", num_samples=104_014),
    ],
)

LS_OTHER_CONFIG = DatasetConfig(
    name="librispeech-other",
    base="librispeech",
    subset="other",
    splits=[
        DatasetSplitConfig(name="train.500", num_samples=148_688),
    ],
)

PS_CLEAN_CONFIG = DatasetConfig(
    name="peoplespeech",
    path="fixie-ai/peoples_speech",
    subset="clean",
    splits=[
        DatasetSplitConfig(name="train", num_samples=1_000_000),
        DatasetSplitConfig(name="validation", num_samples=10_000),
    ],
)

VP_EN_CONFIG = DatasetConfig(
    name="voxpopuli-en",
    path="facebook/voxpopuli",
    subset="en",
    splits=[
        DatasetSplitConfig(name="train", num_samples=1_000_000),
        DatasetSplitConfig(name="validation", num_samples=10_000),
    ],
    assistant_template="{{raw_text}}",
    transcript_template="{{raw_text}}",
)

CV_EN_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-en-transcription",
    base="commonvoice-en",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_AR_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-ar-transcription",
    base="commonvoice-ar",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_DE_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-de-transcription",
    base="commonvoice-de",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_ES_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-es-transcription",
    base="commonvoice-es",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_FR_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-fr-transcription",
    base="commonvoice-fr",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_IT_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-it-transcription",
    base="commonvoice-it",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_JA_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-ja-transcription",
    base="commonvoice-ja",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_PT_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-pt-transcription",
    base="commonvoice-pt",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_RU_TRANS_CONFIG = DatasetConfig(
    name="commonvoice-ru-transcription",
    base="commonvoice-ru",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)

LS_CLEAN_TRANS_CONFIG = DatasetConfig(
    name="librispeech-clean-transcription",
    base="librispeech-clean",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
LS_OTHER_TRANS_CONFIG = DatasetConfig(
    name="librispeech-other-transcription",
    base="librispeech-other",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)

PS_CLEAN_TRANS_CONFIG = DatasetConfig(
    name="peoplespeech-clean-transcription",
    base="peoplespeech",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)

CV_EN_CONT_CONFIG = DatasetConfig(
    name="commonvoice-en-continuation",
    base="commonvoice-en",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_AR_CONT_CONFIG = DatasetConfig(
    name="commonvoice-ar-continuation",
    base="commonvoice-ar",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_DE_CONT_CONFIG = DatasetConfig(
    name="commonvoice-de-continuation",
    base="commonvoice-de",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_ES_CONT_CONFIG = DatasetConfig(
    name="commonvoice-es-continuation",
    base="commonvoice-es",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_FR_CONT_CONFIG = DatasetConfig(
    name="commonvoice-fr-continuation",
    base="commonvoice-fr",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_IT_CONT_CONFIG = DatasetConfig(
    name="commonvoice-it-continuation",
    base="commonvoice-it",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_JA_CONT_CONFIG = DatasetConfig(
    name="commonvoice-ja-continuation",
    base="commonvoice-ja",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_PT_CONT_CONFIG = DatasetConfig(
    name="commonvoice-pt-continuation",
    base="commonvoice-pt",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_RU_CONT_CONFIG = DatasetConfig(
    name="commonvoice-ru-continuation",
    base="commonvoice-ru",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)

LS_CLEAN_CONT_CONFIG = DatasetConfig(
    name="librispeech-clean-continuation",
    base="librispeech-clean",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
LS_OTHER_CONT_CONFIG = DatasetConfig(
    name="librispeech-other-continuation",
    base="librispeech-other",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)

PS_CLEAN_CONT_CONFIG = DatasetConfig(
    name="peoplespeech-clean-continuation",
    base="peoplespeech",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)

INTERNAL_DATASETS = [
    BOOLQ_CONFIG,
    CV_BASE_CONFIG,
    CV_EN_CONFIG,
    CV_AR_CONFIG,
    CV_DE_CONFIG,
    CV_ES_CONFIG,
    CV_FR_CONFIG,
    CV_IT_CONFIG,
    CV_JA_CONFIG,
    CV_PT_CONFIG,
    CV_RU_CONFIG,
    CV_EN_TRANS_CONFIG,
    CV_AR_TRANS_CONFIG,
    CV_DE_TRANS_CONFIG,
    CV_ES_TRANS_CONFIG,
    CV_FR_TRANS_CONFIG,
    CV_IT_TRANS_CONFIG,
    CV_JA_TRANS_CONFIG,
    CV_PT_TRANS_CONFIG,
    CV_RU_TRANS_CONFIG,
    CV_EN_CONT_CONFIG,
    CV_AR_CONT_CONFIG,
    CV_DE_CONT_CONFIG,
    CV_ES_CONT_CONFIG,
    CV_FR_CONT_CONFIG,
    CV_IT_CONT_CONFIG,
    CV_JA_CONT_CONFIG,
    CV_PT_CONT_CONFIG,
    CV_RU_CONT_CONFIG,
    GS_XL_CONFIG,
    LS_BASE_CONFIG,
    LS_CLEAN_CONFIG,
    LS_OTHER_CONFIG,
    LS_CLEAN_TRANS_CONFIG,
    LS_OTHER_TRANS_CONFIG,
    LS_CLEAN_CONT_CONFIG,
    LS_OTHER_CONT_CONFIG,
    PS_CLEAN_CONFIG,
    PS_CLEAN_TRANS_CONFIG,
    PS_CLEAN_CONT_CONFIG,
    VP_EN_CONFIG,
]
DATASET_MAP: Dict[str, DatasetConfig] = {}


def register_datasets(datasets: List[DatasetConfig]):
    for config in datasets:
        name = config.name
        assert name not in DATASET_MAP, f"Dataset {name} already registered"
        DATASET_MAP[name] = config


def unregister_datasets(datasets: List[str]):
    for name in datasets:
        del DATASET_MAP[name]


def _merge_configs(configs: List[DatasetConfig]) -> DatasetConfig:
    merged_config = dataclasses.replace(configs[0])
    for config in configs[1:]:
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if field.name != "base" and value is not None:
                merged_config = dataclasses.replace(
                    merged_config, **{field.name: value}
                )
    return merged_config


def create_dataset(name: str, args: VoiceDatasetArgs) -> SizedIterableDataset:
    assert name in DATASET_MAP, f"Unknown dataset: {name}"
    # Make a list of configs from root->base.
    configs: List[DatasetConfig] = []
    temp: Optional[str] = name
    while temp:
        config = DATASET_MAP[temp]
        configs.insert(0, config)
        temp = config.base
    # Set the root config, and then apply any non-None overrides from the subclasses.
    merged_config = _merge_configs(configs)
    # Sanity check.
    if not merged_config.path:
        raise ValueError(f"Dataset {name} has no path")
    if not merged_config.splits:
        raise ValueError(f"Dataset {name} has no splits")
    return GenericDataset(args, merged_config)


register_datasets(INTERNAL_DATASETS)
