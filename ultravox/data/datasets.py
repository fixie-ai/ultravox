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
from typing import Any, Callable, Dict, List, Optional, Sequence

import datasets
import jinja2
import librosa
import numpy as np
import requests
import soundfile as sf
import streaming as mds
import torch
import torch.nn.functional as F
import transformers
from torch.utils import data

from ultravox.data import dataset_config
from ultravox.data import text_proc

SAMPLE_RATE = 16000

TRANSCRIBE_INPUT_TASK = "transcribe_input"
TRANSCRIBE_OUTPUT_TASK = "transcribe_output"
ANSWER_TASK = "answer"

TRANSCRIBE_PROMPTS = [
    # from Gazelle
    "Transcribe\n<|audio|>",
    "Transcribe exactly what is said here\n<|audio|>",
    "Repeat exactly what is written here: <|audio|>",
    "Write exactly what was said: <|audio|>",
    "First listen to the clip. Then, transcribe exactly what is said. <|audio|>",
    # from https://arxiv.org/pdf/2402.08846
    "Transcribe speech to text: <|audio|>",
    # from GPT-4
    "Capture every word from the audio verbatim\n<|audio|>",
    "Convert speech to text from audio\n<|audio|>",
    "Listen and transcribe the complete text from audio\n<|audio|>",
    "Record in writing what is spoken in audio\n<|audio|>",
    "Transcribe the spoken words from audio with exact wording and punctuation\n<|audio|>",
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


class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclasses.dataclass
class VoiceDatasetArgs:
    data_dir: Optional[str] = None
    prompt: Optional[str] = None
    """A specific prompt to use for the dataset."""
    num_prompts: int = 1
    """If `prompt` is not set, the number of canned prompts to use."""
    include_audio: bool = True
    """Whether to include audio in the samples."""
    include_context: bool = True
    """Whether to include additional textual context from the dataset to the prompt."""
    max_context_length: int = 1500
    """Maximum length of context to include in the prompt. Otherwise, skip the sample."""
    shuffle: bool = False
    """Whether to shuffle the dataset."""
    shuffle_seed: int = 42
    """Seed for shuffling the dataset."""
    max_audio_duration_secs: Optional[float] = None
    """Whether to skip samples with audio longer than this duration."""
    use_mds: bool = False
    """Whether to load the dataset from GCP (using MDS) or Hugging Face."""
    mds_batch_size: int = 32
    """Batch size for MDS."""
    split: DatasetSplit = DatasetSplit.TRAIN
    """Which split of the dataset to use."""

    def __post_init__(self):
        if isinstance(self.split, str):
            self.split = DatasetSplit(self.split.lower())


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
    def __len__(self) -> int:
        pass


class VoiceDataset(SizedIterableDataset):
    """
    Base class for streaming voice datasets.
    Wraps a Hugging Face dataset or MDS-formatted dataset from GCP.
    """

    BASE_AUDIO_COLUMNS = ["audio"]

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__()
        self._args = args
        self._session: Optional[requests.Session] = None
        self._rng = np.random.default_rng(self._args.shuffle_seed)
        self._weight = 1.0  # the default weight for the dataset

    def _init_dataset(self, dataset: data.Dataset, estimated_length: int = 1) -> None:
        self._dataset = dataset
        # Only required when using epochs when training dataset.
        self._estimated_length = estimated_length

    @property
    def weight(self) -> float:
        return self._weight

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
            )
            for column_name in self.BASE_AUDIO_COLUMNS:
                dataset = dataset.cast_column(
                    column_name, datasets.Audio(sampling_rate=SAMPLE_RATE)
                )
            if shuffle:
                dataset = dataset.shuffle(seed=self._args.shuffle_seed)
            return dataset

    def __iter__(self):
        actual_length = 0
        for _, row in enumerate(self._dataset):
            sample = self._get_sample(row)
            if sample is not None:
                if (
                    self._args.max_audio_duration_secs is None
                    or sample.audio is None
                    or sample.audio.shape[-1] / SAMPLE_RATE
                    <= self._args.max_audio_duration_secs
                ):
                    yield sample
            actual_length += 1
            # If len(dataset) == 0 most likely the dataset is a validation dataset,
            # or the training is using max_steps instead of num_epochs.
            if actual_length > len(self) and len(self) > 1:
                warnings.warn(
                    f"The estimated length {self._estimated_length} has been exceeded for type {type(self._dataset)}. Make sure to update."
                )

        if actual_length != len(self) and len(self) > 1:
            warnings.warn(
                f"Mismatch between estimated length ({self._estimated_length}) and actual length ({actual_length}) for dataset of type {type(self._dataset)}. Make sure to update."
            )

    def __len__(self) -> int:
        return int(self._estimated_length * self._weight)

    @abc.abstractmethod
    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        """
        Converts a row from the dataset into a VoiceSample.
        Returns None if the sample should be skipped.
        """

    def _choice(self, prompts: List[str]) -> str:
        return self._rng.choice(prompts[: self._args.num_prompts])

    def _get_answer_prompt(self) -> str:
        if self._args.prompt:
            return self._args.prompt
        return self._choice(ANSWER_PROMPTS)

    def _get_transcribe_prompt(self) -> str:
        if self._args.prompt:
            return self._args.prompt
        return self._choice(TRANSCRIBE_PROMPTS)

    def _get_answer_messages(
        self, question: str, answer: str, context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        prompt = self._get_answer_prompt() if self._args.include_audio else question
        user_content = f"{context}\n\n{prompt}" if context else prompt
        return _get_messages(user_content, answer)

    def _get_transcribe_messages(self, text: str) -> List[Dict[str, str]]:
        prompt = self._get_transcribe_prompt()
        if not self._args.include_audio:
            prompt = prompt.replace("<|audio|>", text)
        return _get_messages(prompt, text)

    def _get_audio(
        self, row: transformers.BatchFeature, column_name: str = "audio"
    ) -> np.ndarray:
        if column_name not in self.BASE_AUDIO_COLUMNS:
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

    def _get_transcribe_sample(
        self,
        row: transformers.BatchFeature,
        tcol: str = "text",
        tproc: Optional[Callable[[str], str]] = None,
    ) -> Optional[VoiceSample]:
        try:
            text = tproc(row[tcol]) if tproc else row[tcol]
        except text_proc.FormatASRError:
            return None
        return self._make_sample(
            self._get_transcribe_messages(text),
            self._get_audio(row),
            audio_transcript=text,
        )

    def _make_sample(
        self,
        messages: List[Dict[str, str]],
        audio: np.ndarray,
        audio_transcript: Optional[str] = None,
    ) -> VoiceSample:
        if not self._args.include_audio:
            return VoiceSample(messages)
        return VoiceSample(messages, audio, audio_transcript=audio_transcript)


class LibriSpeechDummyDataset(VoiceDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            streaming=False,  # not supported by the dummy dataset
        )
        self._init_dataset(dataset, 73)

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        return self._get_transcribe_sample(row, tproc=text_proc.format_asr_text)


# Making EmptyDataset a SizedIterableDataset to be compatible with using epochs during training.
class EmptyDataset(SizedIterableDataset):
    def __init__(self, estimated_length: int = 1) -> None:
        self._estimated_length = estimated_length

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return self._estimated_length


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
        super().__init__(args)
        dataset = datasets.load_dataset(
            "json",
            "anyinstruct",
            data_files="https://huggingface.co/datasets/fnlp/AnyInstruct/resolve/main/speech_conv/metadata.jsonl",
            split="train",
        )
        dataset = dataset.train_test_split(
            test_size=0.01, seed=args.shuffle_seed, shuffle=True
        )
        dataset = dataset["train" if args.split == DatasetSplit.TRAIN else "test"]
        # TODO: make num_shards configurable if need be
        dataset = dataset.to_iterable_dataset(num_shards=16)
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

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        chat = row["chat"]
        return self._make_sample(
            self._get_answer_messages(chat[0]["message"], chat[1]["message"]),
            self._load_anyinstruct_audio(chat[0]["speech"]),
            audio_transcript=chat[0]["message"],
        )


class AnyInstructInputDataset(AnyInstructDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        audio_transcript = row["chat"][0]["message"]
        return self._make_sample(
            self._get_transcribe_messages(audio_transcript),
            self._load_anyinstruct_audio(row["chat"][0]["speech"]),
            audio_transcript=audio_transcript,
        )


class AnyInstructOutputDataset(AnyInstructDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        audio_transcript = row["chat"][1]["message"]
        return self._make_sample(
            self._get_transcribe_messages(audio_transcript),
            self._load_anyinstruct_audio(row["chat"][1]["speech"]),
            audio_transcript=audio_transcript,
        )


class BoolQDataset(VoiceDataset):
    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "fixie-ai/boolq-audio", split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        question = row["question"]
        answer = "True" if row["answer"] else "False"
        context = row["passage"] if self._args.include_context else None
        return self._make_sample(
            self._get_answer_messages(question, answer, context),
            self._get_audio(row),
            audio_transcript=row["question"],
        )


class BoolQInputDataset(BoolQDataset):
    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        return self._get_transcribe_sample(row, tcol="question")


class QAVoiceDatasetMixin(VoiceDataset):
    SEPARATORS = ["\n\n", "\n", "\n----\n"]
    QUERY_PREFIX = ["Question: ", "Question:\n", "Q: ", "Q:\n", "Query: ", "Query:\n"]
    CONTEXT_PREFIX = [
        "Passage: ",
        "Passage:\n",
        "Context: ",
        "Context:\n",
        "Background: ",
        "Background:\n",
    ]
    ANSWER_PREFIX = [
        "Answer: ",
        "A: ",
        "",
        "The answer is: ",
        "Result: ",
        "Conclusion: ",
    ]
    # In most cases there is no extra prompt-suffix needed
    PROMPT_SUFFIXES = [""]

    # TODO: combine `_get_query_prompt` and `_get_answer_messages` into a single method
    # and use this mixin for all non-ASR datasets.
    def _get_query_prompt(self, question_str: str, context: str) -> Optional[str]:
        """
        Creates a random prompt for a QA sample with a passage and question.

        Example prompt:
            Passage: {context}
            Question: {question}
            {optional-prompt-suffix}
        """
        if len(context) > self._args.max_context_length:
            # Skip samples with long context
            return None

        if self._args.prompt:
            prompt = self._args.prompt
        else:
            prompt = self._choice(self.PROMPT_SUFFIXES)

        # Separate either with 1 or 2 newlines
        separator = self._choice(self.SEPARATORS)

        query_prompt = self._choice(self.QUERY_PREFIX)
        question = "<|audio|>" if self._args.include_audio else question_str
        prompt = f"{query_prompt}{question}{separator}{prompt}"

        if self._args.include_context:
            context_prompt = self._choice(self.CONTEXT_PREFIX)
            prompt = f"{context_prompt}{context}{separator}{prompt}"

        return prompt.strip()


class BoolQWithExtendedAnswerDataset(BoolQDataset, QAVoiceDatasetMixin):
    """
    A version of BoolQ that includes the context in the prompt and a longer explanation in the answer.
    """

    PROMPT_SUFFIXES = [
        "Provide a short explanation, then respond with True/False on the last line",
        "Explain briefly, concluding with True/False on a new line."
        "Write a quick explanation, and finish with True/False on the last line"
        "Summarize in a few words, and end with True/False on a new line."
        "Give a brief explanation first, then answer with True/False on the final line",
        "Start with a concise explanation, and end with a True/False response on the last line.",
        "Explain briefly and follow up with True/False at the end",
        "Write a short explanation, then state True/False on a new line.",
        "First, offer a brief explanation, and then reply with True/False at the end.",
        "Present a concise explanation, ending with a True/False answer on the final line",
        "Start with a brief explanation, and then answer with True/False at the end.",
    ]

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        """
        Example conversation:
            <|user|> Passage: {context}
            Question: {question}
            Provide a short explanation, then respond with True/False on the last line
            <|assistant|> {short_explanation}
            Answer: {answer}
        """
        answer = "True" if row["answer"] else "False"
        answer_prompt = self._choice(self.ANSWER_PREFIX)
        user_message = self._get_query_prompt(
            question_str=row["question"], context=row["passage"]
        )
        if user_message is None:
            # Skips samples with long context
            return None

        messages = _get_messages(
            user_message, f"{row['explanation']}\n{answer_prompt}{answer}"
        )

        return self._make_sample(
            messages, self._get_audio(row), audio_transcript=row["question"]
        )


class HeySQuADHumanDataset(QAVoiceDatasetMixin):
    """
    HeySQuAD is a large-scale Spoken Question Answering (SQA) dataset which includes 76k human-spoken questions,
    97k machine-generated questions, and their corresponding textual answers from the SQuAD QA dataset.
    https://arxiv.org/abs/2304.13689

    This dataset is the human-spoken version of HeySQuAD.
    """

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "fixie-ai/HeySQuAD_human", split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        """
        Example conversation
            <|user|> Context: {context}
            Question: {question}
            <|assistant|> {answer}
        """
        if row["is_impossible"] or not row["answers"]:
            # Skip samples with no answer
            return None

        prompt = self._get_query_prompt(
            question_str=row["question"], context=row["context"]
        )
        if prompt is None:
            # Skips samples with long context
            return None

        messages = _get_messages(prompt, row["answers"][0]["text"])
        return self._make_sample(
            messages, self._get_audio(row), audio_transcript=row["question"]
        )


class SlueSQA5Dataset(QAVoiceDatasetMixin):
    """
    SLUE-SQA-5 Dataset contains question texts, question audio, answer text, document text, and document audio from these datasets:
      * SQuAD1.1 (for questions whose question_id starts with 'squad-')
      * Natural Questions (for questions whose question_id starts with 'nq-')
      * TriviaQA (for questions whose question_id starts with 'triviaqa-')
    The following datasets are supposed to be included, but I haven't found them everywhere:
      * WebQuestions (for questions whose question_id starts with 'wq-')
      * CuratedTREC (for questions whose question_id starts with 'trec-')
      * Spoken Wikipedia


    Splits: train, validation, test, verified_test
    """

    BASE_AUDIO_COLUMNS = ["question_audio", "document_audio"]

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "asapp/slue-phase-2", "sqa5", split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        """
        Example conversation
            <|user|> Context: {context}
            Question: {question}
            <|assistant|> {answer}
        """
        prompt = self._get_query_prompt(
            question_str=row["raw_question_text"], context=row["raw_document_text"]
        )
        if prompt is None:
            # Skips samples with long context
            return None

        messages = _get_messages(prompt, row["answer_spans"]["answer"][0])
        return self._make_sample(
            messages,
            self._get_audio(row, "question_audio"),
            audio_transcript=row["raw_question_text"],
        )


# TODO: this dataset can be replaced with GenericVoiceDataset and will be removed/updated in the future.
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

    def _get_sample(self, row: transformers.BatchFeature) -> Optional[VoiceSample]:
        return self._get_transcribe_sample(row, tproc=text_proc.format_asr_text)


# TODO: this dataset can be replaced with GenericVoiceDataset and will be removed/updated in the future.
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

    def _get_sample(self, row) -> Optional[VoiceSample]:
        return self._get_transcribe_sample(row, tproc=text_proc.format_asr_text)


# TODO: this dataset can be replaced with GenericVoiceDataset and will be removed/updated in the future.
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

    def _get_sample(self, row) -> Optional[VoiceSample]:
        return self._get_transcribe_sample(row, tcol="raw_text")


# TODO: this dataset can be replaced with GenericVoiceDataset and will be removed/updated in the future.
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

    def __init__(self, args: VoiceDatasetArgs, lang: str = "en") -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "mozilla-foundation/common_voice_16_1", lang, split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, row) -> Optional[VoiceSample]:
        return self._get_transcribe_sample(row, tcol="sentence")


# TODO: this dataset can be replaced with GenericVoiceDataset and will be removed/updated in the future.
class CoVoST2Dataset(VoiceDataset):
    """
    CoVoST 2 is a large-scale multilingual speech translation corpus covering translations from 21 languages into English
    and from English into 15 languages. The dataset is created using Mozilla's open-source Common Voice 4 database of
    crowdsourced voice recordings. There are 2,900 hours of speech represented in the corpus.

    The original Hugging Face dataset link: https://huggingface.co/datasets/facebook/covost2
    Since this dataset requires audio files to be downloaded separately, a new dataset is created with the audio files:
    https://huggingface.co/datasets/fixie-ai/covost2

    Due to the scale of the dataset and the audio files being repeated, only a portion of the dataset was converted.
    See [this issue](https://github.com/fixie-ai/ultravox/issues/50) for more information.

    Supported subsets (En -> X):
        'en_de', 'en_tr', 'en_fa', 'en_sv-SE', 'en_mn', 'en_zh-CN', 'en_cy',
        'en_ca', 'en_sl', 'en_et', 'en_id', 'en_ar', 'en_ta', 'en_lv', 'en_ja'
    Supported subsets (X -> En):
        'fr_en', 'zh-CN_en', 'es_en'
    """

    CODE_TO_LANG = {
        "en": "English",
        "de": "German",
        "tr": "Turkish",
        "fa": "Persian",
        "sv-SE": "Swedish",
        "mn": "Mongolian",
        "zh-CN": "Chinese",
        "cy": "Welsh",
        "ca": "Catalan",
        "sl": "Slovenian",
        "et": "Estonian",
        "id": "Indonesian",
        "ar": "Arabic",
        "ta": "Tamil",
        "lv": "Latvian",
        "ja": "Japanese",
        "fr": "French",
        "es": "Spanish",
    }

    # We currently don't use this dataset for training, so mainly the first prompt it ever used.
    # The "no explanation" part is important, specially for evaluations, but it's not repeated
    # in all prompts to avoid being too repetitive in training.
    TRANSLATE_PROMPTS = [
        "Translate the following into {target}, without any explanation: <|audio|>",
        "Translate the following into {target} language, no explanation needed: <|audio|>",
        "Please convert the following into {target}. Be concise.\n<|audio|>",
        "Could you translate this to {target} language? No commentary necessary.\n<|audio|>",
        "Translate the text below to {target}.\n<|audio|>",
        "Translate the subsequent text into {target} language. <|audio|>",
        "Can you translate this into the {target} language?\n<|audio|>",
        "Transform the following to {target}: <|audio|>",
    ]

    def __init__(self, args: VoiceDatasetArgs, subset: str) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "fixie-ai/covost2", subset, split=args.split.value
        )
        langs = subset.split("_")
        assert len(langs) == 2, f"Invalid subset: {subset}"
        self.source_lang = self.CODE_TO_LANG[langs[0]]
        self.target_lang = self.CODE_TO_LANG[langs[1]]
        self._init_dataset(dataset)

    def _get_sample(self, row) -> VoiceSample:
        prompt = self._choice(self.TRANSLATE_PROMPTS).format(target=self.target_lang)

        transcript = row["sentence"]
        translation = row["translation"]
        if not self._args.include_audio:
            prompt = prompt.replace("<|audio|>", transcript)

        return self._make_sample(
            _get_messages(prompt, translation),
            self._get_audio(row),
            audio_transcript=transcript,
        )


# TODO: this dataset can be replaced with GenericVoiceDataset and will be removed/updated in the future.
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

    def _get_sample(self, row) -> Optional[VoiceSample]:
        return self._get_transcribe_sample(row, tcol="text")


class SodaDataset(VoiceDataset):
    BASE_AUDIO_COLUMNS = ["audio_second_last_turn"]

    SYS_PROMPTS = [
        "Follow the flow of the conversation and respond just like a human would in the same situation.",
        "Engage in the conversation naturally, responding as a human would.",
        "Follow the dialogue and reply like a person in that situation.",
        "Participate in the chat and answer as if you were a human.",
        "Interact smoothly and respond just like a person would.",
        "Stay in the moment and reply as a human would in the conversation.",
        "Flow with the discussion and respond naturally, as a person would.",
        "Keep the dialogue going and answer like a human would.",
        "Follow along and reply in a way a person would in the chat.",
        "Stay engaged in the conversation and respond like a human.",
        "Maintain the flow of the chat and answer just as a person would.",
    ]

    def __init__(self, args: VoiceDatasetArgs) -> None:
        super().__init__(args)
        dataset = self._load_audio_dataset(
            "fixie-ai/soda-audio", split=args.split.value
        )
        self._init_dataset(dataset)

    def _get_sample(self, row) -> VoiceSample:
        turns = row["dialogue"]
        # Make sure the last turn is the assistant's
        roles = ["user", "assistant"] if len(turns) % 2 == 0 else ["assistant", "user"]

        sys_prompt = self._choice(self.SYS_PROMPTS)

        messages = _get_messages(*turns[:-1], sys_prompt=sys_prompt)

        messages[-1]["content"] = row["alt_last_turn"]
        if self._args.include_audio:
            messages[-2]["content"] = "<|audio|>"

        return self._make_sample(
            messages,
            audio=self._get_audio(row, "audio_second_last_turn"),
            audio_transcript=turns[-2],
        )


class GenericVoiceDataset(VoiceDataset):
    def __init__(
        self, args: VoiceDatasetArgs, config: dataset_config.DataDictConfig
    ) -> None:
        super().__init__(args)
        dataset = datasets.concatenate_datasets(
            [
                self._load_audio_dataset(
                    config.path,
                    name=config.name,
                    split=s,
                    streaming=config.streaming,
                    shuffle=False,
                )
                for s in config.splits
            ]
        )
        # shuffling is only supported on huggingface datasets for now, not MDS
        if self._args.shuffle:
            dataset = dataset.shuffle(seed=self._args.shuffle_seed)

        if config.num_samples:
            dataset = Range(dataset, config.num_samples, config.total_samples)

        self._weight = config.weight

        self.user_template = config.user_template
        self.assistant_template = config.assistant_template
        self.transcript_template = config.transcript_template

        super()._init_dataset(dataset, config.total_samples)

    def _get_sample(self, row) -> VoiceSample:
        try:
            user_content = jinja2.Template(
                self.user_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc, dataset=self)
            assistant_content = jinja2.Template(
                self.assistant_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc, dataset=self)
            transcript = jinja2.Template(
                self.transcript_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc, dataset=self)
        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"user_template: {self.user_template}")
            print(f"assistant_template: {self.assistant_template}")
            print(f"transcript_template: {self.transcript_template}")
            print(f"sample keys: {list(row.keys())}")
            raise ValueError(
                f"Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e

        return self._make_sample(
            _get_messages(user_content, assistant_content),
            self._get_audio(row),
            audio_transcript=transcript,
        )


def create_dataset(name: str, args: VoiceDatasetArgs) -> SizedIterableDataset:
    DATASET_MAP: Dict[str, Any] = {
        "anyinstruct": AnyInstructAnswerDataset,
        "anyinstruct_in": AnyInstructInputDataset,
        "anyinstruct_out": AnyInstructOutputDataset,
        "boolq": BoolQDataset,
        "boolq_in": BoolQInputDataset,
        "boolq_extended": BoolQWithExtendedAnswerDataset,
        "heysquad_human": HeySQuADHumanDataset,
        "slue_sqa5": SlueSQA5Dataset,
        "gigaspeech": GigaSpeechDataset,
        "librispeech": LibriSpeechDataset,
        "voxpopuli": VoxPopuliDataset,
        "commonvoice": CommonVoiceDataset,
        "covost2": CoVoST2Dataset,
        "peoplespeech": PeopleSpeechDataset,
        "soda": SodaDataset,
        "dummy": LibriSpeechDummyDataset,
    }
    if isinstance(name, dataset_config.DataDictConfig):
        return GenericVoiceDataset(args, name)
    else:
        name, *ext = name.split(":")
        return DATASET_MAP[name](args, *ext)


class StopStrategy(str, Enum):
    FIRST_EXHAUSTED = "first_exhausted"
    LAST_EXHAUSTED = "last_exhausted"
    NEVER_STOP = "never_stop"


class InterleaveDataset(SizedIterableDataset):
    """Interleaves multiple IterableDataset objects based on normalized weights."""

    def __init__(
        self,
        datasets: Sequence[SizedIterableDataset],
        stop_strategy: StopStrategy = StopStrategy.LAST_EXHAUSTED,
        seed: Optional[int] = 42,
        static: bool = False,
        using_epochs: bool = False,
    ) -> None:
        """
        Args:
            datasets: A list of SizedIterableDataset objects.
            stop_strategy: Strategy for stopping iteration.
            seed: Optional seed for reproducibility.
            static: If true, the datasets are interleaved in a static order with equal weights.
        """
        self._datasets = datasets
        self._rng = np.random.default_rng(seed)
        self._static = static

        self._stop_strategy = stop_strategy
        self._using_epochs = using_epochs
        if not self._using_epochs:
            weights = [int(getattr(ds, "weight", 1) * len(ds)) for ds in datasets]
        else:
            weights = [getattr(ds, "weight", 1) for ds in datasets]

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

    # Only used when using_epochs is True
    def __len__(self) -> int:
        # TODO: Implement the length method for different stop strategies
        return sum(int(getattr(ds, "weight", 1) * len(ds)) for ds in self._datasets)


class Dataproc(SizedIterableDataset):
    """Base class to preprocess a dataset of VoiceSamples."""

    def __init__(self, dataset: SizedIterableDataset) -> None:
        self._dataset = dataset

    @abc.abstractmethod
    def _process(self, sample: VoiceSample) -> Dict[str, Any]:
        pass

    def __iter__(self):
        return (self._process(sample) for sample in self._dataset)

    def __len__(self) -> int:
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

    def __len__(self) -> int:
        return (
            self._num_samples
            if self._num_samples is not None
            else self._estimated_length
        )
