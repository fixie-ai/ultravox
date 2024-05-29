import abc
import base64
import dataclasses
import enum
import io
import logging
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

import datasets
import librosa
import numpy as np
import requests
import soundfile as sf
import streaming as mds
import torch
import transformers
from torch.utils import data

from ultravox.data import text_proc

SAMPLE_RATE = 16000

TRANSCRIBE_INPUT_TASK = "transcribe_input"
TRANSCRIBE_OUTPUT_TASK = "transcribe_output"
ANSWER_TASK = "answer"

TRANSCRIBE_PROMPTS = [
    # from Gazelle
    "Transcribe <|audio|>",
    "Transcribe exactly what is said here <|audio|>",
    "Repeat exactly what is written here: <|audio|>",
    "Write exactly what was said: <|audio|>",
    "First listen to the clip. Then, transcribe exactly what is said. <|audio|>",
    # from https://arxiv.org/pdf/2402.08846
    "Transcribe speech to text: <|audio|>",
    # from GPT-4
    "Capture every word from <|audio|> verbatim",
    "Convert speech to text from <|audio|>",
    "Listen and transcribe the complete text from <|audio|>",
    "Record in writing what is spoken in <|audio|>",
    "Transcribe the spoken words from <|audio|> with exact wording and punctuation",
]
ANSWER_PROMPTS = [
    # from Gazelle
    "Listen to <|audio|> and respond to it",
    "Listen and respond: <|audio|>",
    "Respond to <|audio|>",
    "Respond to the user <|audio|>",
    "<|audio|>",
    "<|audio|>",  # repeated to emphasize not needing a prompt for Q&A tasks
    "Respond to this question: \n<|audio|>",
    "Continue the conversation after <|audio|>",
    "First listen to the clip: <|audio|>\n How would you respond?",
    "<|audio|> - respond",
    "<|audio|>\n Respond to the question",
]

# TODO(juberti): set these in the environment so they don't need to be hard-coded here.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "fixie-training"


# Silence the spurious warnings coming from the MosaicML streaming library.
logging.getLogger("streaming.base.dataset").setLevel(logging.ERROR)


@dataclasses.dataclass
class DataCollatorForSeq2SeqWithAudio(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features, *args, **kwargs):
        audio_features = [f.pop("audio_values") for f in features]
        batch = super().__call__(features, *args, **kwargs)
        batch["audio_values"] = torch.nn.utils.rnn.pad_sequence(
            audio_features, batch_first=True
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
            assert (
                self.audio.dtype == np.float32
            ), f"Unexpected audio dtype: {self.audio.dtype}"
            assert self.audio.ndim == 1, f"Unexpected audio shape: {self.audio.shape}"

    messages: List[Dict[str, str]]
    """List of messages, each with a "role" and "content" field."""
    audio: Optional[np.ndarray] = None
    """Audio data as float32 PCM @ `sample_rate`."""
    sample_rate: int = SAMPLE_RATE
    """Audio sample rate in Hz."""
    audio_transcript: Optional[str] = None
    """For evaluations, the known transcript of the audio."""


class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclasses.dataclass
class VoiceDatasetArgs:
    data_dir: Optional[str] = None
    num_prompts: int = 1
    shuffle: bool = False
    shuffle_seed: int = 42
    max_audio_duration_secs: Optional[float] = None
    use_mds: bool = False
    mds_batch_size: int = 32
    split: DatasetSplit = DatasetSplit.TRAIN

    def __post_init__(self):
        if isinstance(self.split, str):
            self.split = DatasetSplit(self.split.lower())


class VoiceDataset(abc.ABC, data.IterableDataset):
    """
    Base class for streaming voice datasets.
    Wraps a Hugging Face dataset or MDS-formatted dataset from GCP.
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__()
        self._args = args
        self._session: Optional[requests.Session] = None

    def _init_dataset(self, dataset: data.Dataset) -> None:
        self._dataset = dataset

    def _load_audio_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        shuffle: Optional[bool] = None,
        streaming: bool = True,
    ) -> data.Dataset:
        logging.info(f"Loading dataset {path} {name} {split} {shuffle} {streaming}")
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
            dataset = datasets.load_dataset(
                path, name, split=split, trust_remote_code=True, streaming=streaming
            ).cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))
            if shuffle:
                dataset = dataset.shuffle(seed=self._args.shuffle_seed)
            return dataset

    def __iter__(self):
        for i, row in enumerate(self._dataset):
            sample = self._get_sample(i, row)
            if (
                self._args.max_audio_duration_secs is None
                or sample.audio.shape[-1] / SAMPLE_RATE
                <= self._args.max_audio_duration_secs
            ):
                yield sample

    @abc.abstractmethod
    def _get_sample(self, idx: int, row: transformers.BatchFeature) -> VoiceSample:
        pass

    def _get_answer_prompt(self, idx: int) -> str:
        prompt_idx = idx % min(self._args.num_prompts, len(ANSWER_PROMPTS))
        return ANSWER_PROMPTS[prompt_idx]

    def _get_transcribe_prompt(self, idx: int) -> str:
        prompt_idx = idx % min(self._args.num_prompts, len(TRANSCRIBE_PROMPTS))
        return TRANSCRIBE_PROMPTS[prompt_idx]

    def _get_answer_messages(self, idx: int, text: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": self._get_answer_prompt(idx)},
            {"role": "assistant", "content": text},
        ]

    def _get_transcribe_messages(self, idx: int, text: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": self._get_transcribe_prompt(idx)},
            {"role": "assistant", "content": text},
        ]

    def _get_audio(self, row: transformers.BatchFeature) -> np.ndarray:
        # Hugging Face datasets have an Audio object, with array and sampling_rate fields.
        # For MDS, this object is flattened into audio_array and audio_sampling_rate fields.
        if "audio" in row:
            audio = row["audio"]["array"]
            sampling_rate = row["audio"]["sampling_rate"]
        elif "audio_array" in row:
            audio = row["audio_array"]
            sampling_rate = row["audio_sampling_rate"]
        else:
            raise ValueError("No audio field found in row.")
        assert sampling_rate == SAMPLE_RATE
        return audio

    def _get_transcribe_sample(
        self,
        idx: int,
        row: transformers.BatchFeature,
        tcol: str = "text",
        tproc: Optional[Callable[[str], str]] = None,
    ) -> VoiceSample:
        text = tproc(row[tcol]) if tproc else row[tcol]
        return VoiceSample(
            self._get_transcribe_messages(idx, text),
            self._get_audio(row),
            audio_transcript=text,
        )

    def _load_audio(self, base_url: str, folder: str, filename: str) -> np.ndarray:
        if self._args.data_dir:
            audio_path = f"{self._args.data_dir}/{folder}/{filename}"
            audio = audio_from_file(audio_path)
        else:
            url = f"{base_url}/{filename}"  # hack for GCS bucket naming
            if self._session is None:
                self._session = requests.Session()
            response = self._session.get(url)
            response.raise_for_status()
            audio = audio_from_buf(response.content)
        return audio


class LibriSpeechDummyDataset(VoiceDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            streaming=False,  # not supported by the dummy dataset
        )
        self._init_dataset(dataset)

    def _get_sample(self, idx: int, row: transformers.BatchFeature) -> VoiceSample:
        return self._get_transcribe_sample(idx, row, tproc=text_proc.format_asr_text)


class EmptyDataset(data.IterableDataset):
    def __iter__(self):
        return iter([])


class AnyInstructDataset(VoiceDataset):
    """
    Metadata file format:
    {"chat": [
        {"role": "USER", "message": "Write a sentence based on this summary: iraqi embassy in jakarta removes saddam hussein 's photo", "speech": "chunk_00000/0001.mp3"},
        {"role": "AnyGPT", "message": "The building in Jakarta where people from Iraq work, took down a picture of a man named Saddam Hussein.", "speech": "chunk_00000/0002.mp3"}
    ]}
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        # TODO(juberti): convert to MDS
        # The last 7 samples are missing audio files, so we exclude them.
        NUM_SAMPLES = 108193 - 7
        super().__init__(args)
        dataset = (
            datasets.load_dataset(
                "json",
                "anyinstruct",
                data_files="https://huggingface.co/datasets/fnlp/AnyInstruct/resolve/main/speech_conv/metadata.jsonl",
                split="train",
            ).select(range(NUM_SAMPLES))
            # TODO: make num_shards configurable if need be
            .to_iterable_dataset(num_shards=16)
        )
        if args.shuffle:
            dataset = dataset.shuffle(seed=args.shuffle_seed)
        self._init_dataset(dataset)

    def _load_anyinstruct_audio(self, filename: str):
        return super()._load_audio(
            "https://storage.googleapis.com/train-anyinstruct-speechconv-v1",
            "anyinstruct/speech",
            filename,
        )


class AnyInstructAnswerDataset(AnyInstructDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)

    def _get_sample(self, idx: int, row: transformers.BatchFeature) -> VoiceSample:
        return VoiceSample(
            self._get_answer_messages(idx, row["chat"][1]["message"]),
            self._load_anyinstruct_audio(row["chat"][0]["speech"]),
            audio_transcript=row["chat"][0]["message"],
        )


class AnyInstructInputDataset(AnyInstructDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)

    def _get_sample(self, idx: int, row: transformers.BatchFeature) -> VoiceSample:
        audio_transcript = row["chat"][0]["message"]
        return VoiceSample(
            self._get_transcribe_messages(idx, audio_transcript),
            self._load_anyinstruct_audio(row["chat"][0]["speech"]),
            audio_transcript=audio_transcript,
        )


class AnyInstructOutputDataset(AnyInstructDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)

    def _get_sample(self, idx: int, row: transformers.BatchFeature) -> VoiceSample:
        audio_transcript = row["chat"][1]["message"]
        return VoiceSample(
            self._get_transcribe_messages(idx, audio_transcript),
            self._load_anyinstruct_audio(row["chat"][1]["speech"]),
            audio_transcript=audio_transcript,
        )


class BoolQDataset(VoiceDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        assert (
            args.split == DatasetSplit.VALIDATION
        ), f"BoolQ is only for validation, but got split={args.split}"
        super().__init__(args)
        dataset = self._load_audio_dataset("fixie-ai/boolq-audio", split="train")
        self._init_dataset(dataset)

    def _get_sample(self, idx: int, row: transformers.BatchFeature) -> VoiceSample:
        return VoiceSample(
            self._get_answer_messages(idx, "True" if row["answer"] else "False"),
            self._get_audio(row),
            audio_transcript=row["question"],
        )


class BoolQInputDataset(BoolQDataset):
    def _get_sample(self, idx: int, row: transformers.BatchFeature) -> VoiceSample:
        audio_transcript = str(row["question"])
        return VoiceSample(
            self._get_transcribe_messages(idx, audio_transcript),
            self._get_audio(row),
        )


class LibriSpeechDataset(VoiceDataset):
    """
    LibriSpeech is a corpus of approximately 1000 hours of 16kHz read
    English speech. The data is derived from read audiobooks from the
    LibriVox project. A simple automatic procedure was used to select
    the audio in the first two sets to be, on average, of higher
    recording quality and with accents closer to US English.
    https://huggingface.co/datasets/librispeech_asr
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        # TODO(juberti): convert to MDS, in a way that preserves the same
        # concatenation of the three splits. MDS can interleave but not
        # concatenate, it seems.
        super().__init__(args)
        ds: Any
        if args.split == DatasetSplit.VALIDATION:
            ds = self._load_audio_dataset("librispeech_asr", split="validation.clean")
        else:
            splits = ["train.clean.100", "train.clean.360", "train.other.500"]
            ds = datasets.concatenate_datasets(
                [
                    self._load_audio_dataset("librispeech_asr", split=s, shuffle=False)
                    for s in splits
                ]
            )
        if self._args.shuffle:
            ds = ds.shuffle(seed=self._args.shuffle_seed)
        self._init_dataset(ds)

    def _get_sample(self, idx: int, row: transformers.BatchFeature) -> VoiceSample:
        return self._get_transcribe_sample(idx, row, tproc=text_proc.format_asr_text)


class GigaSpeechDataset(VoiceDataset):
    """
    GigaSpeech is an evolving, multi-domain English speech recognition corpus
    with 10,000 hours of high quality labeled audio suitable for supervised training.
    "s" split is 250 hours. Non-commercial use only.
    https://huggingface.co/datasets/speechcolab/gigaspeech
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "speechcolab/gigaspeech", "xl", split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, idx, row) -> VoiceSample:
        return self._get_transcribe_sample(idx, row, tproc=text_proc.format_asr_text)


class VoxPopuliDataset(VoiceDataset):
    """
    VoxPopuli is a large-scale multilingual speech corpus for representation learning,
    semi-supervised learning and interpretation.
    "en" split is 543 hours.
    https://huggingface.co/datasets/facebook/voxpopuli
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "facebook/voxpopuli", "en", split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, idx, row) -> VoiceSample:
        return self._get_transcribe_sample(idx, row, tcol="raw_text")


class CommonVoiceDataset(VoiceDataset):
    """
    The Common Voice dataset consists of a unique MP3 and corresponding text file
    https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1
    Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 1090061
    })
    NOTE: requires HF login
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "mozilla-foundation/common_voice_16_1", "en", split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, idx, row) -> VoiceSample:
        return self._get_transcribe_sample(idx, row, tcol="sentence")


class PeopleSpeechDataset(VoiceDataset):
    """
    The People's Speech Dataset is among the world's largest English speech
    recognition corpus. It includes 30,000+ hours of transcribed speech in
    English languages with a diverse set of speakers.
    https://huggingface.co/datasets/MLCommons/peoples_speech
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "MLCommons/peoples_speech", "clean", split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, idx, row) -> VoiceSample:
        return self._get_transcribe_sample(idx, row, tcol="text")


def create_dataset(name: str, args: VoiceDatasetArgs) -> data.IterableDataset:
    DATASET_MAP: Dict[str, Any] = {
        "anyinstruct": AnyInstructAnswerDataset,
        "anyinstruct_in": AnyInstructInputDataset,
        "anyinstruct_out": AnyInstructOutputDataset,
        "boolq": BoolQDataset,
        "boolq_in": BoolQInputDataset,
        "gigaspeech": GigaSpeechDataset,
        "librispeech": LibriSpeechDataset,
        "voxpopuli": VoxPopuliDataset,
        "commonvoice": CommonVoiceDataset,
        "peoplespeech": PeopleSpeechDataset,
        "dummy": LibriSpeechDummyDataset,
    }
    return DATASET_MAP[name](args)


class InterleaveDataset(data.IterableDataset):
    """Interleaves multiple IterableDataset objects."""

    def __init__(
        self, datasets: Sequence[data.IterableDataset], repeat: bool = False
    ) -> None:
        """
        Args:
            datasets: a list of IterableDataset objects
            repeat: whether to repeat the datasets indefinitely.
                This matters most when the datasets have different lengths.
                Let's say you have two datasets, A and B which have 5 and 3 samples respectively.

                `repeat=False`: [A0, B0, A1, B1, A2, B2, A3, A4]
                `repeat=True` : [A0, B0, A1, B1, A2, B2, A3, B0, A4, B1, A0, ...]

                NOTE: with `repeat=True`, `__iter__` never stops.
        """
        super().__init__()
        self._datasets = datasets
        self._repeat = repeat

    def __iter__(self):
        iters = [iter(ds) for ds in self._datasets]
        iter_index = 0

        while len(iters):
            it = iters[iter_index]
            try:
                val = next(it)
                iter_index = (iter_index + 1) % len(iters)
                yield val
            except StopIteration:
                if not self._repeat:
                    iters.pop(iter_index)
                    iter_index %= max(1, len(iters))
                else:
                    iters[iter_index] = iter(self._datasets[iter_index])


class Dataproc(abc.ABC, data.IterableDataset):
    """Base class to preprocess a dataset of VoiceSamples."""

    def __init__(self, dataset: data.IterableDataset) -> None:
        self._dataset = dataset

    @abc.abstractmethod
    def _process(self, sample: VoiceSample) -> Dict[str, Any]:
        pass

    def __iter__(self):
        return (self._process(sample) for sample in self._dataset)


class Range(data.IterableDataset):
    """Limits the number of samples from another dataset."""

    def __init__(
        self, dataset: data.IterableDataset, num_samples: Optional[int] = None
    ) -> None:
        self._dataset = dataset
        self._num_samples = num_samples

    def __iter__(self):
        for i, sample in enumerate(self._dataset):
            if self._num_samples is not None and i >= self._num_samples:
                break
            yield sample
